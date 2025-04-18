import edu.princeton.cs.algs4.IndexMinPQ;

import java.util.*;


class DijkstraShortestPath {
    private final double[][] distTo;
    private final int[][] edgeToX;
    private final int[][] edgeToY;
    private final ImageProcessor processor;
    private int centerX, centerY;
    private int searchRadius;

    public DijkstraShortestPath(ImageProcessor processor,int radius) {
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

        for (int x = Math.max(0, centerX - searchRadius); x < Math.min(processor.getWidth(), centerX + searchRadius); x++) {
            for (int y = Math.max(0, centerY - searchRadius); y < Math.min(processor.getHeight(), centerY + searchRadius); y++) {
                if (distanceSq(x, y, centerX, centerY) > searchRadius * searchRadius) {
                    continue;
                }
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

            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx;
                    int ny = y + dy;

                    if (distanceSq(x, y, centerX, centerY) > searchRadius * searchRadius) continue;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

                    double cost = 1.0 / (1 + processor.getGradient(nx, ny));
                    if (distTo[x][y] + cost < distTo[nx][ny]) {
                        distTo[nx][ny] = distTo[x][y] + cost;
                        edgeToX[nx][ny] = x;
                        edgeToY[nx][ny] = y;
                        if (pq.contains(nx * height + ny)) {
                            pq.decreaseKey(nx * height + ny, distTo[nx][ny]);
                        } else {
                            pq.insert(nx * height + ny, distTo[nx][ny]);
                        }
                    }
                }
            }
        }
    }

    public List<Point> getPathTo(int endX, int endY) {
        LinkedList<Point> path = new LinkedList<>();
        int x = endX;
        int y = endY;

        while (x != -1 && y != -1) {
            path.addFirst(new Point(x, y));

            if (distanceSq(x, y, centerX, centerY) >= searchRadius * searchRadius) {
                break;
            }

            int px = edgeToX[x][y];
            int py = edgeToY[x][y];
            x = px;
            y = py;
        }
        return path;
    }

    private int distanceSq(int x1, int y1, int x2, int y2) {
        int dx = x1 - x2;
        int dy = y1 - y2;
        return dx * dx + dy * dy;
    }
}