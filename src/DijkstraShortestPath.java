import edu.princeton.cs.algs4.IndexMinPQ;

import java.util.*;


class DijkstraShortestPath {
    private final double[][] distTo;
    private final int[][] edgeToX;
    private final int[][] edgeToY;
    private final ImageProcessor processor;
    private int centerX, centerY;
    private int searchRadius;
    private final int[][] dirs = {
            {-1, -1, 141}, // √2 ≈ 1.414 → 141 (整型存储避免浮点运算)
            {-1, 0, 100},  // 1.0 → 100
            {-1, 1, 141},
            {0, -1, 100},
            {0, 1, 100},
            {1, -1, 141},
            {1, 0, 100},
            {1, 1, 141}
    };

    public DijkstraShortestPath(ImageProcessor processor, int radius) {
        this.processor = processor;
        int width = processor.getWidth();
        int height = processor.getHeight();
        this.distTo = new double[width][height];
        this.edgeToX = new int[width][height];
        this.edgeToY = new int[width][height];
        this.searchRadius = radius;

        for (int x = 0; x < width; x++) {
            Arrays.fill(distTo[x], Double.POSITIVE_INFINITY);
            Arrays.fill(edgeToX[x], -1);
            Arrays.fill(edgeToY[x], -1);
        }
    }

    public void computeFrom(int startX, int startY) {
        int width = processor.getWidth();
        int height = processor.getHeight();
        this.centerX = startX;
        this.centerY = startY;
        int radiusSq = searchRadius * searchRadius;

        // Optimized initialization loop for circular area
        for (int x = Math.max(0, centerX - searchRadius); x < Math.min(width, centerX + searchRadius + 1); x++) {
            int dx = x - centerX;
            int dxSq = dx * dx;
            if (dxSq > radiusSq) continue;
            int remainingSq = radiusSq - dxSq;
            if (remainingSq < 0) continue;
            int maxDy = (int) Math.sqrt(remainingSq);
            int yStart = Math.max(0, centerY - maxDy);
            int yEnd = Math.min(height, centerY + maxDy + 1);
            for (int y = yStart; y < yEnd; y++) {
                distTo[x][y] = Double.POSITIVE_INFINITY;
                edgeToX[x][y] = -1;
                edgeToY[x][y] = -1;
            }
        }

        IndexMinPQ<Double> pq = new IndexMinPQ<>(width * height);
        distTo[startX][startY] = 0;
        pq.insert(startX * height + startY, 0.0);

        while (!pq.isEmpty()) {
            int v = pq.delMin();
            int x = v / height;
            int y = v % height;

            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];

                // Inlined distance check and boundary check
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                int deltaX = nx - centerX;
                int deltaY = ny - centerY;
                if (deltaX * deltaX + deltaY * deltaY > radiusSq) continue;

                // Cache gradient value
                double cost = processor.getCost(nx, ny) * (dir[2]/100.0);

                if (distTo[x][y] + cost < distTo[nx][ny]) {
                    distTo[nx][ny] = distTo[x][y] + cost;
                    edgeToX[nx][ny] = x;
                    edgeToY[nx][ny] = y;
                    int index = nx * height + ny;
                    if (pq.contains(index)) {
                        pq.decreaseKey(index, distTo[nx][ny]);
                    } else {
                        pq.insert(index, distTo[nx][ny]);
                    }
                }
            }
        }
    }

    public List<Point> getPathTo(int endX, int endY) {
        LinkedList<Point> path = new LinkedList<>();
        int x = endX;
        int y = endY;
        int radiusSq = searchRadius * searchRadius;

        while (x != -1 && y != -1) {
            path.addFirst(new Point(x, y));

            int deltaX = x - centerX;
            int deltaY = y - centerY;
            if (deltaX * deltaX + deltaY * deltaY >= radiusSq) {
                break;
            }

            int px = edgeToX[x][y];
            int py = edgeToY[x][y];
            x = px;
            y = py;
        }
        return smoothPath(interpolateWithCatmullRom(path));
        //return (interpolateWithCatmullRom(path));
        //return smoothPath(path);
        //return path;
    }

    private LinkedList<Point> interpolateWithCatmullRom(LinkedList<Point> originalPath) {
        LinkedList<Point> interpolatedPath = new LinkedList<>();
        if (originalPath.size() < 2) return originalPath;

        int n = originalPath.size();
        double step = 0.5;

        // 添加第一个控制点（保持起点不变）
        interpolatedPath.add(originalPath.getFirst());

        for (int i = 0; i < n - 1; i++) {
            Point p0 = (i > 0) ? originalPath.get(i - 1) : originalPath.get(i);
            Point p1 = originalPath.get(i);
            Point p2 = originalPath.get(i + 1);
            Point p3 = (i + 2 < n) ? originalPath.get(i + 2) : p2;

            // 生成当前段插值点
            for (double t = step; t < 1.0; t += step) {
                double x = calculateSplineValue(t, p0.x, p1.x, p2.x, p3.x);
                double y = calculateSplineValue(t, p0.y, p1.y, p2.y, p3.y);

                Point interpolatedPoint = new Point((int) Math.round(x), (int) Math.round(y));

                // 去重处理
                if (interpolatedPath.isEmpty() ||
                        !interpolatedPath.getLast().equals(interpolatedPoint)) {
                    interpolatedPath.add(interpolatedPoint);
                }
            }
        }

        // 确保包含最后一个点
        Point last = originalPath.getLast();
        if (!interpolatedPath.getLast().equals(last)) {
            interpolatedPath.add(last);
        }

        return interpolatedPath;
    }

    private double calculateSplineValue(double t, double p0, double p1, double p2, double p3) {
        double t2 = t * t;
        double t3 = t2 * t;
        return 0.5 * (
                (-p0 + 3*p1 - 3*p2 + p3) * t3 +
                        (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                        (-p0 + p2) * t +
                        2*p1
        );
    }

    private List<Point> smoothPath(List<Point> path) {
        // 使用滑动窗口平均
        List<Point> smoothed = new ArrayList<>();
        int window = 1;
        for (int i=0; i<path.size(); i++) {
            int sumX = 0, sumY = 0;
            int count = 0;
            for (int j=-window; j<=window; j++) {
                if (i+j >=0 && i+j < path.size()) {
                    Point p = path.get(i+j);
                    sumX += p.x;
                    sumY += p.y;
                    count++;
                }
            }
            smoothed.add(new Point(sumX/count, sumY/count));
        }
        return smoothed;
    }
}