import java.nio.ByteBuffer;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.stream.IntStream;
import java.util.concurrent.atomic.AtomicInteger;

class ImageProcessor {
    private final ByteBuffer imageData;
    private final int width;
    private final int height;
    private final double[][] gradient;
    private static final int THRESHOLD = 1000;
    private static final int CACHE_LINE_SIZE = 64;
    private static final int BLOCK_SIZE = 16; // 减小块大小以增加并行度
    private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors() * 2; // 增加线程数
    private static final ForkJoinPool pool = new ForkJoinPool(NUM_THREADS);
    private final AtomicInteger processedBlocks = new AtomicInteger(0);

    public ImageProcessor(ByteBuffer data, int width, int height) {
        this.imageData = data.asReadOnlyBuffer();
        this.width = width;
        this.height = height;
        this.gradient = new double[width][height];
        calculateGradients();
    }

    private void calculateGradients() {
        final int[][] sobelX = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
        final int[][] sobelY = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
        final double edgeValue = 1e3;

        int numBlocksX = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int numBlocksY = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // 使用更细粒度的任务分配
        pool.invoke(new BlockGradientTask(0, numBlocksX, 0, numBlocksY,
                sobelX, sobelY, edgeValue));

        processBorders(edgeValue);
    }

    private class BlockGradientTask extends RecursiveAction {
        private final int startBlockX, endBlockX, startBlockY, endBlockY;
        private final int[][] sobelX, sobelY;
        private final double edgeValue;

        BlockGradientTask(int startBlockX, int endBlockX, int startBlockY, int endBlockY,
                int[][] sobelX, int[][] sobelY, double edgeValue) {
            this.startBlockX = startBlockX;
            this.endBlockX = endBlockX;
            this.startBlockY = startBlockY;
            this.endBlockY = endBlockY;
            this.sobelX = sobelX;
            this.sobelY = sobelY;
            this.edgeValue = edgeValue;
        }

        @Override
        protected void compute() {
            int totalBlocks = (endBlockX - startBlockX) * (endBlockY - startBlockY);
            if (totalBlocks <= 1) {
                computeBlock();
            } else {
                int midBlockX = (startBlockX + endBlockX) >>> 1;
                int midBlockY = (startBlockY + endBlockY) >>> 1;

                invokeAll(
                        new BlockGradientTask(startBlockX, midBlockX, startBlockY, midBlockY,
                                sobelX, sobelY, edgeValue),
                        new BlockGradientTask(midBlockX, endBlockX, startBlockY, midBlockY,
                                sobelX, sobelY, edgeValue),
                        new BlockGradientTask(startBlockX, midBlockX, midBlockY, endBlockY,
                                sobelX, sobelY, edgeValue),
                        new BlockGradientTask(midBlockX, endBlockX, midBlockY, endBlockY,
                                sobelX, sobelY, edgeValue));
            }
        }

        private void computeBlock() {
            for (int blockX = startBlockX; blockX < endBlockX; blockX++) {
                for (int blockY = startBlockY; blockY < endBlockY; blockY++) {
                    int startX = blockX * BLOCK_SIZE;
                    int startY = blockY * BLOCK_SIZE;
                    int endX = Math.min(startX + BLOCK_SIZE, width);
                    int endY = Math.min(startY + BLOCK_SIZE, height);

                    // 预加载缓存行
                    for (int x = startX; x < endX; x += CACHE_LINE_SIZE) {
                        // 预加载当前行的像素数据
                        int[] rowCache = new int[CACHE_LINE_SIZE];
                        for (int i = 0; i < CACHE_LINE_SIZE && x + i < endX; i++) {
                            rowCache[i] = getPixel(x + i, startY);
                        }

                        for (int y = startY; y < endY; y++) {
                            if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
                                handleBorderPixel(x, y, edgeValue);
                                continue;
                            }

                            // 使用局部变量减少内存访问
                            double dx = 0, dy = 0;
                            int p00 = getPixel(x - 1, y - 1);
                            int p01 = getPixel(x, y - 1);
                            int p02 = getPixel(x + 1, y - 1);
                            int p10 = getPixel(x - 1, y);
                            int p12 = getPixel(x + 1, y);
                            int p20 = getPixel(x - 1, y + 1);
                            int p21 = getPixel(x, y + 1);
                            int p22 = getPixel(x + 1, y + 1);

                            // 展开计算
                            dx = p00 * sobelX[0][0] + p01 * sobelX[0][1] + p02 * sobelX[0][2] +
                                    p10 * sobelX[1][0] + p12 * sobelX[1][2] +
                                    p20 * sobelX[2][0] + p21 * sobelX[2][1] + p22 * sobelX[2][2];

                            dy = p00 * sobelY[0][0] + p01 * sobelY[0][1] + p02 * sobelY[0][2] +
                                    p10 * sobelY[1][0] + p12 * sobelY[1][2] +
                                    p20 * sobelY[2][0] + p21 * sobelY[2][1] + p22 * sobelY[2][2];

                            gradient[x][y] = Math.sqrt(dx * dx + dy * dy);
                        }
                    }

                    // 更新进度
                    processedBlocks.incrementAndGet();
                }
            }
        }
    }

    private void handleBorderPixel(int x, int y, double value) {
        if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
            gradient[x][y] = value;
        } else {
            gradient[x][y] = 0;
        }
    }

    private void processBorders(double value) {
        // 增加并行度
        IntStream.range(1, width - 1).parallel().forEach(x -> {
            gradient[x][0] = value;
            gradient[x][height - 1] = value;
        });

        IntStream.range(1, height - 1).parallel().forEach(y -> {
            gradient[0][y] = value;
            gradient[width - 1][y] = value;
        });
    }

    private int getPixel(int x, int y) {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return 0;
        return imageData.get(y * width + x) & 0xFF;
    }

    public double getGradient(int x, int y) {
        return gradient[x][y];
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }
}