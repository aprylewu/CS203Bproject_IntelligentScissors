import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

class ImageProcessor {
    private final byte[] pixelArray;
    private final int width;
    private final int height;
    private double[][] laplacian;
    private double[][][] gradientDir;
    private double[] directionCostMap;
    private double[] fusedCostArray;
    private final int[][] gray;
    private final ForkJoinPool pool;
    private static final int PARALLELISM = 14;
    private static final double FINE_WEIGHT = 0.95;
    private static final double COARSE_WEIGHT_1 = 0.48;
    private static final double COARSE_WEIGHT_2 = 0.02;
    private static final double LAPLACE_THRESHOLD = 50.0;
    private static final double Z_WEIGHT = 0.43;
    private static final double D_WEIGHT = 0.43;
    private static final double G_WEIGHT = 0.14;
    private static final double LBP_WEIGHT = 0.02;
    private static final double GABOR_WEIGHT = 0.03;
    private static final double SHARPEN_AMOUNT = 1.5;
    private static final double SHARPEN_SIGMA = 1.5;
    private static final int SHARPEN_RADIUS = 4;
    private static final double EdgeValue = 0.15;
    private static final double GABOR_THETA_STEP = Math.PI / 4;
    private static final int GABOR_SCALES = 2;
    private static final double[] GABOR_LAMBDA = {2.0, 4.0};
    private static final double GABOR_SIGMA = 0.5 * GABOR_LAMBDA[0];
    private static final int THRESHOLD = 20;

    public ImageProcessor(ByteBuffer data, int width, int height) {

        int[][] gray1;
        this.width = width;
        this.height = height;
        this.pixelArray = new byte[width * height];
        data.get(pixelArray);
        this.pool = new ForkJoinPool(PARALLELISM);
        gray1 = precomputeGrayPixels();

        gray1 = applySharpen(gray1, SHARPEN_AMOUNT, SHARPEN_SIGMA, SHARPEN_RADIUS);

        // 原尺度特征
        this.gray = gray1;
        double[][][] fineGradientDir = new double[width][height][2];
        double[] fineGradient = computeGradients(gray, width, height, fineGradientDir);
        int[] fineLBP = computeLBP(gray, width, height);
        double[] fineGabor = computeGaborFeatures(gray, width, height);

        // 粗尺度特征1
        int scaleFactor_1 = 2;
        int[][] coarseGray = downsample(gray, scaleFactor_1);
        int coarseWidth = width / scaleFactor_1;
        int coarseHeight = height / scaleFactor_1;
        double[][][] coarseGradientDir1 = new double[coarseWidth][coarseHeight][2];
        double[] coarseGradient = computeGradients(coarseGray, coarseWidth, coarseHeight, coarseGradientDir1);
        int[] coarseLBP = computeLBP(coarseGray, coarseWidth, coarseHeight);
        double[] coarseGabor1 = computeGaborFeatures(coarseGray, coarseWidth, coarseHeight);


        // 上采样粗尺度特征1
        double[][][] upsampledDir1 = upsampleDirection(coarseGradientDir1, coarseWidth, coarseHeight, scaleFactor_1);
        double[] upsampledGradient = upsample(coarseGradient, coarseWidth, coarseHeight, scaleFactor_1);
        int[] upsampledLBP = upsampleLBP(coarseLBP, coarseWidth, coarseHeight, scaleFactor_1);
        double[] upsampledGabor1 = upsample(coarseGabor1, coarseWidth, coarseHeight, scaleFactor_1);

        // 粗尺度特征2
        int scaleFactor_2 = 4;
        int[][] coarseGray2 = downsample(gray, scaleFactor_2);
        int coarseWidth2 = width / scaleFactor_2;
        int coarseHeight2 = height / scaleFactor_2;
        double[][][] coarseGradientDir2 = new double[coarseWidth2][coarseHeight2][2];
        double[] coarseGradient2 = computeGradients(coarseGray2, coarseWidth2, coarseHeight2, coarseGradientDir2);
        int[] coarseLBP2 = computeLBP(coarseGray2, coarseWidth2, coarseHeight2);
        double[] coarseGabor2 = computeGaborFeatures(coarseGray2, coarseWidth2, coarseHeight2);

        // 上采样粗尺度特征2
        double[][][] upsampledDir2 = upsampleDirection(coarseGradientDir2, coarseWidth2, coarseHeight2, scaleFactor_2);
        double[] upsampledGradient2 = upsample(coarseGradient2, coarseWidth2, coarseHeight2, scaleFactor_2);
        int[] upsampledLBP2 = upsampleLBP(coarseLBP2, coarseWidth2, coarseHeight2, scaleFactor_2);
        double[] upsampledGabor2 = upsample(coarseGabor2, coarseWidth2, coarseHeight2, scaleFactor_2);

        computeLaplacian();
        // 多尺度方向融合
        fuseGradientDirections(fineGradientDir, upsampledDir1, upsampledDir2);
        computeDirectionCost();

        // 多尺度特征融合
        fuseFeatures(fineGradient, upsampledGradient, upsampledGradient2,
                fineLBP, upsampledLBP, upsampledLBP2,
                fineGabor, upsampledGabor1, upsampledGabor2);
    }

    private int[][] precomputeGrayPixels() {
        int[][] gray = new int[width][height];
        pool.invoke(new GrayComputeTask(0, height, gray));
        return gray;
    }

    private class GrayComputeTask extends RecursiveAction {
        private final int startY, endY;
        private final int[][] gray;

        GrayComputeTask(int startY, int endY, int[][] gray) {
            this.startY = startY;
            this.endY = endY;
            this.gray = gray;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    for (int x = 0; x < width; x++) {
                        gray[x][y] = pixelArray[y * width + x] & 0xFF;
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new GrayComputeTask(startY, mid, gray),
                        new GrayComputeTask(mid, endY, gray)
                );
            }
        }
    }

    private int[] computeLBP(int[][] gray, int w, int h) {
        int[] lbp = new int[w * h];
        pool.invoke(new LBPComputeTask(gray, 0, h, w, h, lbp));
        return lbp;
    }

    private class LBPComputeTask extends RecursiveAction {
        private final int startY, endY, width, height;
        private final int[][] gray;
        private final int[] lbp;

        LBPComputeTask(int[][] gray, int startY, int endY, int width, int height, int[] lbp) {
            this.gray = gray;
            this.startY = Math.max(1, startY);
            this.endY = Math.min(height - 1, endY);
            this.width = width;
            this.height = height;
            this.lbp = lbp;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    for (int x = 1; x < width - 1; x++) {
                        int center = gray[x][y];
                        int code = 0;
                        code |= (gray[x-1][y-1] >= center) ? 0 : 1 << 7;
                        code |= (gray[x][y-1] >= center) ? 0 : 1 << 6;
                        code |= (gray[x+1][y-1] >= center) ? 0 : 1 << 5;
                        code |= (gray[x+1][y] >= center) ? 0 : 1 << 4;
                        code |= (gray[x+1][y+1] >= center) ? 0 : 1 << 3;
                        code |= (gray[x][y+1] >= center) ? 0 : 1 << 2;
                        code |= (gray[x-1][y+1] >= center) ? 0 : 1 << 1;
                        code |= (gray[x-1][y] >= center) ? 0 : 1;
                        lbp[y * width + x] = code;
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new LBPComputeTask(gray, startY, mid, width, height, lbp),
                        new LBPComputeTask(gray, mid, endY, width, height, lbp)
                );
            }
        }
    }

    private int[][] downsample(int[][] original, int scale) {
        int w = original.length;
        int h = original[0].length;
        int newW = w / scale;
        int newH = h / scale;
        int[][] downsampled = new int[newW][newH];
        pool.invoke(new DownsampleTask(original, 0, newH, scale, downsampled));
        return downsampled;
    }

    private class DownsampleTask extends RecursiveAction {
        private final int[][] original;
        private final int startY, endY, scale;
        private final int[][] downsampled;

        DownsampleTask(int[][] original, int startY, int endY, int scale, int[][] downsampled) {
            this.original = original;
            this.startY = startY;
            this.endY = endY;
            this.scale = scale;
            this.downsampled = downsampled;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    for (int x = 0; x < downsampled.length; x++) {
                        int sum = 0, count = 0;
                        for (int dy = 0; dy < scale; dy++) {
                            for (int dx = 0; dx < scale; dx++) {
                                int ox = x * scale + dx;
                                int oy = y * scale + dy;
                                if (ox < original.length && oy < original[0].length) {
                                    sum += original[ox][oy];
                                    count++;
                                }
                            }
                        }
                        downsampled[x][y] = count > 0 ? sum / count : 0;
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new DownsampleTask(original, startY, mid, scale, downsampled),
                        new DownsampleTask(original, mid, endY, scale, downsampled)
                );
            }
        }
    }

    private double[] upsample(double[] coarse, int coarseW, int coarseH, int scale) {
        double[] upsampled = new double[width * height];
        pool.invoke(new UpsampleTask(coarse, coarseW, coarseH, 0, height, scale, upsampled));
        return upsampled;
    }

    private class UpsampleTask extends RecursiveAction {
        private static final int THRESHOLD = 100;
        private final double[] coarse;
        private final int coarseW, coarseH, scale;
        private final int startY, endY;
        private final double[] upsampled;

        UpsampleTask(double[] coarse, int coarseW, int coarseH, int startY, int endY, int scale, double[] upsampled) {
            this.coarse = coarse;
            this.coarseW = coarseW;
            this.coarseH = coarseH;
            this.startY = startY;
            this.endY = endY;
            this.scale = scale;
            this.upsampled = upsampled;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    float cy = (float) y / height * coarseH;
                    int y0 = Math.min((int) cy, coarseH - 1);
                    int y1 = Math.min(y0 + 1, coarseH - 1);
                    float ty = cy - y0;

                    for (int x = 0; x < width; x++) {
                        float cx = (float) x / width * coarseW;
                        int x0 = Math.min((int) cx, coarseW - 1);
                        int x1 = Math.min(x0 + 1, coarseW - 1);
                        float tx = cx - x0;

                        double g00 = coarse[y0 * coarseW + x0];
                        double g10 = coarse[y1 * coarseW + x0];
                        double g01 = coarse[y0 * coarseW + x1];
                        double g11 = coarse[y1 * coarseW + x1];

                        upsampled[y * width + x] =
                                (1 - tx) * (1 - ty) * g00 +
                                        tx * (1 - ty) * g01 +
                                        (1 - tx) * ty * g10 +
                                        tx * ty * g11;
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new UpsampleTask(coarse, coarseW, coarseH, startY, mid, scale, upsampled),
                        new UpsampleTask(coarse, coarseW, coarseH, mid, endY, scale, upsampled)
                );
            }
        }
    }

    private int[] upsampleLBP(int[] coarse, int coarseW, int coarseH, int scale) {
        int[] upsampled = new int[width * height];
        pool.invoke(new UpsampleLBPTask(coarse, coarseW, coarseH, scale, upsampled, 0, height));
        return upsampled;
    }

    private class UpsampleLBPTask extends RecursiveAction {
        private final int[] coarse;
        private final int coarseW, coarseH, scale;
        private final int[] upsampled;
        private final int startY, endY;

        UpsampleLBPTask(int[] coarse, int coarseW, int coarseH,
                        int scale, int[] upsampled, int startY, int endY) {
            this.coarse = coarse;
            this.coarseW = coarseW;
            this.coarseH = coarseH;
            this.scale = scale;
            this.upsampled = upsampled;
            this.startY = startY;
            this.endY = endY;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    int cy = Math.min(y / scale, coarseH - 1);
                    for (int x = 0; x < width; x++) {
                        int cx = Math.min(x / scale, coarseW - 1);
                        upsampled[y * width + x] = coarse[cy * coarseW + cx];
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new UpsampleLBPTask(coarse, coarseW, coarseH, scale, upsampled, startY, mid),
                        new UpsampleLBPTask(coarse, coarseW, coarseH, scale, upsampled, mid, endY)
                );
            }
        }
    }

    public boolean hasZeroCrossing(int x, int y) {
        if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) return false;

        double center = laplacian[x][y];
        // 检查8邻域
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                double neighbor = laplacian[x+dx][y+dy];
                if (center * neighbor < -LAPLACE_THRESHOLD * LAPLACE_THRESHOLD) return true;
            }
        }
        return false;
    }

    private void computeLaplacian() {
        laplacian = new double[width][height];
        int[][] kernel = {
                {1, 1, 1},
                {1, -8, 1},
                {1, 1, 1}
        };

        // 使用并行任务处理
        pool.invoke(new LaplacianComputeTask(1, height - 1, kernel));
    }

    private class LaplacianComputeTask extends RecursiveAction {
        private final int startY, endY;
        private final int[][] kernel;

        LaplacianComputeTask(int startY, int endY, int[][] kernel) {
            this.startY = startY;
            this.endY = endY;
            this.kernel = kernel;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                // 直接计算范围内的行
                for (int y = startY; y < endY; y++) {
                    for (int x = 1; x < width - 1; x++) {
                        double sum = 0;
                        // 应用3x3核
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                sum += gray[x + dx][y + dy] * kernel[dx + 1][dy + 1];
                            }
                        }
                        laplacian[x][y] = sum;
                    }
                }
            } else {
                // 拆分任务
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new LaplacianComputeTask(startY, mid, kernel),
                        new LaplacianComputeTask(mid, endY, kernel)
                );
            }
        }
    }

    private double[] computeGradients(int[][] gray, int w, int h, double[][][] gradientDirs) {
        double[] gradients = new double[w * h];
        pool.invoke(new GradientComputeTask(gray, 0, h, w, h, gradients, gradientDirs));
        return gradients;
    }

    private class GradientComputeTask extends RecursiveAction {
        private final int startY, endY, width, height;
        private final int[][] gray;
        private final double[] gradients;
        private final double[][][] gradientDirs;

        GradientComputeTask(int[][] gray, int startY, int endY, int width, int height,
                            double[] gradients, double[][][] gradientDirs) {
            this.gray = gray;
            this.startY = Math.max(1, startY);
            this.endY = Math.min(height - 1, endY);
            this.width = width;
            this.height = height;
            this.gradients = gradients;
            this.gradientDirs = gradientDirs;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    for (int x = 1; x < width - 1; x++) {
                        // Sobel算子计算梯度
                        double dx = (-gray[x-1][y-1] - 2*gray[x-1][y] - gray[x-1][y+1]
                                + gray[x+1][y-1] + 2*gray[x+1][y] + gray[x+1][y+1]);
                        double dy = (-gray[x-1][y-1] - 2*gray[x][y-1] - gray[x+1][y-1]
                                + gray[x-1][y+1] + 2*gray[x][y+1] + gray[x+1][y+1]);

                        // 计算梯度和方向
                        double magnitude = Math.hypot(dx, dy);
                        gradients[y * width + x] = magnitude;
//                        System.out.println(magnitude);

                        // 存储单位方向向量（Iy, -Ix）
                        if (magnitude > 1e-3) {
                            gradientDirs[x][y][0] = dy / magnitude;  // Iy
                            gradientDirs[x][y][1] = -dx / magnitude; // -Ix

//                            System.out.println(gradientDirs[x][y][0]+" "+gradientDirs[x][y][1]);
                        } else {
                            gradientDirs[x][y][0] = 0;
                            gradientDirs[x][y][1] = 0;
                        }
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new GradientComputeTask(gray, startY, mid, width, height, gradients, gradientDirs),
                        new GradientComputeTask(gray, mid, endY, width, height, gradients, gradientDirs)
                );
            }
        }
    }

    private double[][][] upsampleDirection(double[][][] coarseDir, int coarseW, int coarseH, int scale) {
        double[][][] upsampled = new double[width][height][2];
        pool.invoke(new DirectionUpsampleTask(coarseDir, coarseW, coarseH, 0, height, scale, upsampled));
        return upsampled;
    }

    private class DirectionUpsampleTask extends RecursiveAction {
        private final double[][][] coarseDir;
        private final int coarseW, coarseH, scale;
        private final int startY, endY;
        private final double[][][] upsampled;

        DirectionUpsampleTask(double[][][] coarseDir, int coarseW, int coarseH,
                              int startY, int endY, int scale, double[][][] upsampled) {
            this.coarseDir = coarseDir;
            this.coarseW = coarseW;
            this.coarseH = coarseH;
            this.startY = startY;
            this.endY = endY;
            this.scale = scale;
            this.upsampled = upsampled;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    float cy = (float) y / height * coarseH;
                    int y0 = Math.min((int) cy, coarseH - 1);
                    int y1 = Math.min(y0 + 1, coarseH - 1);
                    float ty = cy - y0;

                    for (int x = 0; x < width; x++) {
                        float cx = (float) x / width * coarseW;
                        int x0 = Math.min((int) cx, coarseW - 1);
                        int x1 = Math.min(x0 + 1, coarseW - 1);
                        float tx = cx - x0;

                        // 双线性插值每个方向分量
                        for (int i = 0; i < 2; i++) {
                            double v00 = coarseDir[x0][y0][i];
                            double v01 = coarseDir[x1][y0][i];
                            double v10 = coarseDir[x0][y1][i];
                            double v11 = coarseDir[x1][y1][i];
                            upsampled[x][y][i] =
                                    (1 - tx) * (1 - ty) * v00 +
                                            tx * (1 - ty) * v01 +
                                            (1 - tx) * ty * v10 +
                                            tx * ty * v11;
                        }
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new DirectionUpsampleTask(coarseDir, coarseW, coarseH, startY, mid, scale, upsampled),
                        new DirectionUpsampleTask(coarseDir, coarseW, coarseH, mid, endY, scale, upsampled)
                );
            }
        }
    }

    private void computeDirectionCost() {
        directionCostMap = new double[width * height];
        pool.invoke(new DirectionCostTask(0, height));
    }

    private class DirectionCostTask extends RecursiveAction {
        private final int startY, endY;

        DirectionCostTask(int startY, int endY) {
            this.startY = Math.max(1, startY);
            this.endY = Math.min(height - 1, endY);
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                int[][] offsets = {{-1, -1}, {-1, 0}, {-1, 1},
                        {0, -1}, {0, 1},
                        {1, -1}, {1, 0}, {1, 1}};

                for (int y = startY; y < endY; y++) {
                    for (int x = 1; x < width - 1; x++) {
                        double totalCost = 0;
                        int validDirections = 0;
                        double[] dpDir = gradientDir[x][y];

                        for (int[] dir : offsets) {
                            int nx = x + dir[0];
                            int ny = y + dir[1];
                            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

                            // 方向调整计算（保持原逻辑）
                            double dx = dir[0];
                            double dy = dir[1];
                            double dotProduct = dpDir[0] * dx + dpDir[1] * dy;
                            if (dotProduct < 0) {
                                dx = -dx;
                                dy = -dy;
                            }

                            double length = Math.hypot(dx, dy);
                            if (length < 1e-3) continue;
                            double unitX = dx / length;
                            double unitY = dy / length;

                            // 点积计算（保持原逻辑）
                            double dp = dpDir[0] * unitX + dpDir[1] * unitY;
                            double[] dqDir = gradientDir[nx][ny];
                            double dq = dqDir[0] * unitX + dqDir[1] * unitY;

                            dp = Math.max(-1.0, Math.min(1.0, dp));
                            dq = Math.max(-1.0, Math.min(1.0, dq));
                            double cost = (Math.acos(dp) + Math.acos(dq)) / Math.PI;

                            double distanceWeight = (dir[0] == 0 || dir[1] == 0) ? 1.0 : 1.0/Math.sqrt(2);
                            totalCost += cost * distanceWeight;
                            validDirections++;
                        }

                        directionCostMap[y * width + x] = validDirections > 0 ?
                                totalCost / validDirections : 1.0;
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new DirectionCostTask(startY, mid),
                        new DirectionCostTask(mid, endY)
                );
            }
        }
    }

    private void fuseGradientDirections(double[][][] fineDir, double[][][] coarseDir1, double[][][] coarseDir2) {
        gradientDir = new double[width][height][2];
        pool.invoke(new FuseGradientDirectionsTask(fineDir, coarseDir1, coarseDir2, 0, height));
    }

    private class FuseGradientDirectionsTask extends RecursiveAction {
        private final double[][][] fineDir, coarseDir1, coarseDir2;
        private final int startY, endY;

        FuseGradientDirectionsTask(double[][][] fineDir,
                                   double[][][] coarseDir1,
                                   double[][][] coarseDir2,
                                   int startY, int endY) {
            this.fineDir = fineDir;
            this.coarseDir1 = coarseDir1;
            this.coarseDir2 = coarseDir2;
            this.startY = startY;
            this.endY = endY;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                // 处理当前分片范围内的像素行
                for (int y = startY; y < endY; y++) {
                    for (int x = 0; x < width; x++) {
                        // 加权融合各尺度方向（与原逻辑保持一致）
                        double dirX = FINE_WEIGHT * fineDir[x][y][0]
                                + COARSE_WEIGHT_1 * coarseDir1[x][y][0]
                                + COARSE_WEIGHT_2 * coarseDir2[x][y][0];
                        double dirY = FINE_WEIGHT * fineDir[x][y][1]
                                + COARSE_WEIGHT_1 * coarseDir1[x][y][1]
                                + COARSE_WEIGHT_2 * coarseDir2[x][y][1];

                        // 归一化方向向量
                        double len = Math.hypot(dirX, dirY);
                        if (len > 1e-3) {
                            gradientDir[x][y][0] = dirX / len;
                            gradientDir[x][y][1] = dirY / len;
                        } else {
                            gradientDir[x][y][0] = 0;
                            gradientDir[x][y][1] = 0;
                        }
                    }
                }
            } else {
                // 拆分任务（典型的分治策略）
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new FuseGradientDirectionsTask(fineDir, coarseDir1, coarseDir2, startY, mid),
                        new FuseGradientDirectionsTask(fineDir, coarseDir1, coarseDir2, mid, endY)
                );
            }
        }
    }

    private double[] computeGaborFeatures(int[][] gray, int w, int h) {
        double[] gaborResponse = new double[w * h];
        pool.invoke(new GaborComputeTask(gray, 0, h, w, h, gaborResponse));
        return gaborResponse;
    }

    private class GaborComputeTask extends RecursiveAction {
        private final int[][] gray;
        private final int startY, endY, width, height;
        private final double[] response;

        GaborComputeTask(int[][] gray, int startY, int endY,
                         int width, int height, double[] response) {
            this.gray = gray;
            this.startY = Math.max(1, startY);
            this.endY = Math.min(height - 1, endY);
            this.width = width;
            this.height = height;
            this.response = response;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                // 生成多方向多尺度Gabor滤波器
                List<double[][]> filters = generateGaborFilters();

                for (int y = startY; y < endY; y++) {
                    for (int x = 1; x < width - 1; x++) {
                        double maxResponse = 0;
                        // 计算所有Gabor滤波器的最大响应
                        for (double[][] kernel : filters) {
                            double sum = 0;
                            // 应用Gabor滤波器
                            for (int ky = -2; ky <= 2; ky++) {
                                for (int kx = -2; kx <= 2; kx++) {
                                    int px = x + kx;
                                    int py = y + ky;
                                    if (px < 0 || px >= width || py < 0 || py >= height)
                                        continue;
                                    sum += gray[px][py] * kernel[kx+2][ky+2];
                                }
                            }
                            maxResponse = Math.max(maxResponse, Math.abs(sum));
                        }
                        response[y * width + x] = maxResponse;
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new GaborComputeTask(gray, startY, mid, width, height, response),
                        new GaborComputeTask(gray, mid, endY, width, height, response)
                );
            }
        }

        // 生成多方向多尺度Gabor滤波器
        private List<double[][]> generateGaborFilters() {
            List<double[][]> filters = new ArrayList<>();
            for (int s = 0; s < GABOR_SCALES; s++) {
                double lambda = GABOR_LAMBDA[s];
                for (double theta = 0; theta < Math.PI; theta += GABOR_THETA_STEP) {
                    double[][] kernel = new double[5][5];
                    double sigma = GABOR_SIGMA * (s + 1);

                    for (int y = -2; y <= 2; y++) {
                        for (int x = -2; x <= 2; x++) {
                            double xx = x * Math.cos(theta) + y * Math.sin(theta);
                            double yy = -x * Math.sin(theta) + y * Math.cos(theta);

                            // Gabor公式
                            double g = Math.exp(-(xx*xx + yy*yy) / (2 * sigma*sigma));
                            g *= Math.cos(2 * Math.PI * xx / lambda);
                            kernel[x+2][y+2] = g;
                        }
                    }
                    filters.add(kernel);
                }
            }
            return filters;
        }
    }

    private void fuseFeatures(double[] fineGrad, double[] coarseGrad, double[] coarseGrad2,
                              int[] fineLBP, int[] coarseLBP, int[] coarseLBP2,
                              double[] fineGabor, double[] coarseGabor1, double[] coarseGabor2) {
        fusedCostArray = new double[width * height];

        // 预计算最大值归一化（保持原有逻辑）
        final double maxFineGrad = Arrays.stream(fineGrad).max().orElse(1);
        final double maxCoarseGrad1 = Arrays.stream(coarseGrad).max().orElse(1);
        final double maxCoarseGrad2 = Arrays.stream(coarseGrad2).max().orElse(1);
        final double maxFineGabor = Arrays.stream(fineGabor).max().orElse(1);
        final double maxCoarseGabor1 = Arrays.stream(coarseGabor1).max().orElse(1);
        final double maxCoarseGabor2 = Arrays.stream(coarseGabor2).max().orElse(1);

        // 并行执行特征融合
        pool.invoke(new FuseFeaturesTask(0, height,
                fineGrad, coarseGrad, coarseGrad2,
                fineLBP, coarseLBP, coarseLBP2,
                fineGabor, coarseGabor1, coarseGabor2,
                maxFineGrad, maxCoarseGrad1, maxCoarseGrad2,
                maxFineGabor, maxCoarseGabor1, maxCoarseGabor2));
    }

    private class FuseFeaturesTask extends RecursiveAction {
        private final int startY, endY;
        private final double[] fineGrad, coarseGrad, coarseGrad2;
        private final int[] fineLBP, coarseLBP, coarseLBP2;
        private final double[] fineGabor, coarseGabor1, coarseGabor2;
        private final double maxFineGrad, maxCoarseGrad1, maxCoarseGrad2;
        private final double maxFineGabor, maxCoarseGabor1, maxCoarseGabor2;

        FuseFeaturesTask(int startY, int endY,
                         double[] fineGrad, double[] coarseGrad, double[] coarseGrad2,
                         int[] fineLBP, int[] coarseLBP, int[] coarseLBP2,
                         double[] fineGabor, double[] coarseGabor1, double[] coarseGabor2,
                         double maxFG, double maxCG1, double maxCG2,
                         double maxFGab, double maxCGab1, double maxCGab2) {
            this.startY = Math.max(startY, 1);
            this.endY = Math.min(endY, height - 1);
            this.fineGrad = fineGrad;
            this.coarseGrad = coarseGrad;
            this.coarseGrad2 = coarseGrad2;
            this.fineLBP = fineLBP;
            this.coarseLBP = coarseLBP;
            this.coarseLBP2 = coarseLBP2;
            this.fineGabor = fineGabor;
            this.coarseGabor1 = coarseGabor1;
            this.coarseGabor2 = coarseGabor2;
            this.maxFineGrad = maxFG;
            this.maxCoarseGrad1 = maxCG1;
            this.maxCoarseGrad2 = maxCG2;
            this.maxFineGabor = maxFGab;
            this.maxCoarseGabor1 = maxCGab1;
            this.maxCoarseGabor2 = maxCGab2;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    for (int x = 0; x < width ; x++) {
                        final int i = y * width + x;

                        // 边界处理（与原始逻辑一致）
                        if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                            fusedCostArray[i] = EdgeValue;
                            continue;
                        }

                        // 梯度成本计算
                        double gradCost = FINE_WEIGHT * (1 - fineGrad[i]/maxFineGrad)
                                + COARSE_WEIGHT_1 * (1 - coarseGrad[i]/maxCoarseGrad1)
                                + COARSE_WEIGHT_2 * (1 - coarseGrad2[i]/maxCoarseGrad2);

                        // LBP成本计算
                        double LBPCost = FINE_WEIGHT * fineLBP[i]
                                + COARSE_WEIGHT_1 * coarseLBP[i]
                                + COARSE_WEIGHT_2 * coarseLBP2[i];

                        // Gabor特征计算
                        double gaborFeature = FINE_WEIGHT * (fineGabor[i]/maxFineGabor)
                                + COARSE_WEIGHT_1 * (coarseGabor1[i]/maxCoarseGabor1)
                                + COARSE_WEIGHT_2 * (coarseGabor2[i]/maxCoarseGabor2);

                        // 零交叉检测（保持原有逻辑）
                        double fZ = hasZeroCrossing(x, y) ? 0 : 1;

                        // 最终成本合成
                        fusedCostArray[i] = Z_WEIGHT * fZ
                                + D_WEIGHT * directionCostMap[i]
                                + G_WEIGHT * gradCost
                                + LBP_WEIGHT * (1 - LBPCost/255.0)
                                + GABOR_WEIGHT * (1 - gaborFeature);
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new FuseFeaturesTask(startY, mid,
                                fineGrad, coarseGrad, coarseGrad2,
                                fineLBP, coarseLBP, coarseLBP2,
                                fineGabor, coarseGabor1, coarseGabor2,
                                maxFineGrad, maxCoarseGrad1, maxCoarseGrad2,
                                maxFineGabor, maxCoarseGabor1, maxCoarseGabor2),
                        new FuseFeaturesTask(mid, endY,
                                fineGrad, coarseGrad, coarseGrad2,
                                fineLBP, coarseLBP, coarseLBP2,
                                fineGabor, coarseGabor1, coarseGabor2,
                                maxFineGrad, maxCoarseGrad1, maxCoarseGrad2,
                                maxFineGabor, maxCoarseGabor1, maxCoarseGabor2)
                );
            }
        }
    }

    private int[][] applyGaussianBlur(int[][] input, double sigma, int radius) {
        double[] kernel = generateGaussianKernel(sigma, radius);
        int[][] temp = horizontalConvolution(input, kernel);
        return verticalConvolution(temp, kernel);
    }

    private double[] generateGaussianKernel(double sigma, int radius) {
        int size = 2 * radius + 1;
        double[] kernel = new double[size];
        double sum = 0.0;
        for (int i = -radius; i <= radius; i++) {
            kernel[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma));
            sum += kernel[i + radius];
        }
        // 归一化
        for (int i = 0; i < size; i++) {
            kernel[i] /= sum;
        }
        return kernel;
    }

    private int[][] horizontalConvolution(int[][] input, double[] kernel) {
        int w = input.length;
        int h = input[0].length;
        int[][] output = new int[w][h];
        int radius = (kernel.length - 1) / 2;
        pool.invoke(new HorizontalConvolutionTask(input, output, 0, h, radius, kernel));
        return output;
    }

    private class HorizontalConvolutionTask extends RecursiveAction {
        private final int[][] input, output;
        private final int startY, endY, radius;
        private final double[] kernel;

        HorizontalConvolutionTask(int[][] input, int[][] output, int startY, int endY, int radius, double[] kernel) {
            this.input = input;
            this.output = output;
            this.startY = startY;
            this.endY = endY;
            this.radius = radius;
            this.kernel = kernel;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    for (int x = 0; x < input.length; x++) {
                        double sum = 0.0;
                        for (int k = -radius; k <= radius; k++) {
                            int px = x + k;
                            // 边界处理：使用最近边缘像素
                            if (px < 0) px = 0;
                            else if (px >= input.length) px = input.length - 1;
                            sum += input[px][y] * kernel[k + radius];
                        }
                        output[x][y] = (int) Math.round(sum);
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new HorizontalConvolutionTask(input, output, startY, mid, radius, kernel),
                        new HorizontalConvolutionTask(input, output, mid, endY, radius, kernel)
                );
            }
        }
    }

    private int[][] verticalConvolution(int[][] input, double[] kernel) {
        int w = input.length;
        int h = input[0].length;
        int[][] output = new int[w][h];
        int radius = (kernel.length - 1) / 2;
        pool.invoke(new VerticalConvolutionTask(input, output, 0, h, radius, kernel));
        return output;
    }

    private class VerticalConvolutionTask extends RecursiveAction {
        private final int[][] input, output;
        private final int startY, endY, radius;
        private final double[] kernel;

        VerticalConvolutionTask(int[][] input, int[][] output, int startY, int endY, int radius, double[] kernel) {
            this.input = input;
            this.output = output;
            this.startY = startY;
            this.endY = endY;
            this.radius = radius;
            this.kernel = kernel;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    for (int x = 0; x < input.length; x++) {
                        double sum = 0.0;
                        for (int k = -radius; k <= radius; k++) {
                            int py = y + k;
                            // 边界处理：使用最近边缘像素
                            if (py < 0) py = 0;
                            else if (py >= input[0].length) py = input[0].length - 1;
                            sum += input[x][py] * kernel[k + radius];
                        }
                        output[x][y] = (int) Math.round(sum);
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new VerticalConvolutionTask(input, output, startY, mid, radius, kernel),
                        new VerticalConvolutionTask(input, output, mid, endY, radius, kernel)
                );
            }
        }
    }

    private int[][] applySharpen(int[][] input, double amount, double sigma, int radius) {
        int[][] blurred = applyGaussianBlur(input, sigma, radius);
        int[][] sharpened = new int[input.length][input[0].length];
        pool.invoke(new SharpenTask(input, blurred, sharpened, 0, input[0].length, amount));
        return sharpened;
    }

    private class SharpenTask extends RecursiveAction {
        private final int[][] input;
        private final int[][] blurred;
        private final int[][] sharpened;
        private final int startY, endY;
        private final double amount;

        SharpenTask(int[][] input, int[][] blurred, int[][] sharpened,
                    int startY, int endY, double amount) {
            this.input = input;
            this.blurred = blurred;
            this.sharpened = sharpened;
            this.startY = startY;
            this.endY = endY;
            this.amount = amount;
        }

        @Override
        protected void compute() {
            if (endY - startY <= THRESHOLD) {
                for (int y = startY; y < endY; y++) {
                    for (int x = 0; x < input.length; x++) {
                        // 非锐化掩模算法
                        int original = input[x][y];
                        int blur = blurred[x][y];
                        int detail = original - blur;
                        int value = (int) Math.round(original + amount * detail);
                        sharpened[x][y] = Math.max(0, Math.min(255, value)); // 钳制值域
                    }
                }
            } else {
                int mid = (startY + endY) >>> 1;
                invokeAll(
                        new SharpenTask(input, blurred, sharpened, startY, mid, amount),
                        new SharpenTask(input, blurred, sharpened, mid, endY, amount)
                );
            }
        }
    }

    public double getCost(int x, int y) {
        return fusedCostArray[y * width + x];
    }
    public double getGray(int x, int y) {
        return gray[x][y];
    }

    public void shutdown() {
        pool.shutdown();
    }

    public int getWidth() { return width; }
    public int getHeight() { return height; }
}