import edu.princeton.cs.algs4.IndexMinPQ;
import org.lwjgl.*;
import org.lwjgl.glfw.*;
import org.lwjgl.opengl.*;
import org.lwjgl.stb.*;
import org.lwjgl.system.*;

import java.awt.*;
import java.io.*;
import java.nio.*;
import java.util.*;
import java.util.List;

import static org.lwjgl.glfw.Callbacks.*;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.stb.STBImage.*;
import static org.lwjgl.system.MemoryStack.*;
import static org.lwjgl.system.MemoryUtil.*;

public class IntelligentScissorsLWJGL {
    // Window and OpenGL
    private long window;
    private int winWidth = 2000;
    private int winHeight = 1125;
    final int BORDER = 100;
    private int texID;

    // Image data
    private int imgWidth;
    private int imgHeight;
    private float scale = 1.0f;
    private float panX = 0.0f;
    private float panY = 0.0f;

    // Algorithm components
    private ImageProcessor processor;
    private DijkstraShortestPath dijkstra;
    private Point seedPoint;
    private List<Point> currentPath = new ArrayList<>();

    private List<List<Point>> savedPaths = new ArrayList<>();
    private List<Point> savedSeedPoints = new ArrayList<>();

    private static final int SEARCH_RADIUS = 256;
    private static final double cutPathRatio = 0.67;
    private static double adaptRadius;

    private boolean isInteractionEnabled = false;

    private Point firstSeedPoint;

    public void run() {
        init();
        loop();
        cleanup();
    }

    private void init() {
        GLFWErrorCallback.createPrint(System.err).set();
        if (!glfwInit())
            throw new IllegalStateException("GLFW init failed");

        // Window configuration
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(winWidth, winHeight, "Intelligent Scissors", NULL, NULL);

        // Input callbacks
        setupCallbacks();

        // OpenGL context
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        GL.createCapabilities();

        // Initialize OpenGL
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glfwShowWindow(window);
        loadDefaultImage();
        centerImage(0);
    }

    private void loop() {
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT);

            // 主动获取鼠标位置并模拟回调
            try (MemoryStack stack = stackPush()) {
                DoubleBuffer xBuf = stack.mallocDouble(1);
                DoubleBuffer yBuf = stack.mallocDouble(1);
                glfwGetCursorPos(window, xBuf, yBuf);
                double x = xBuf.get(0);
                double y = yBuf.get(0);

                // 手动调用回调逻辑（相当于模拟 glfwSetCursorPosCallback）
                if (dijkstra != null && seedPoint != null) {
                    double imgX = (x - panX) / scale;
                    double imgY = (y - panY) / scale;

                    Point rawPoint = new Point((int) imgX, (int) imgY);
                    Point projected = projectToSearchRadius(rawPoint);

                    imgX = projected.x;
                    imgY = projected.y;

                    imgX = Math.max(0, Math.min(imgX, imgWidth - 1));
                    imgY = Math.max(0, Math.min(imgY, imgHeight - 1));

                    if (isValidPoint((int) imgX, (int) imgY)) {
                        if (distanceSq((int) imgX, (int) imgY, seedPoint.x, seedPoint.y) >= SEARCH_RADIUS
                                * SEARCH_RADIUS) {
                            Point edgePoint = findEdgePoint(new Point((int) imgX, (int) imgY));
                            if (edgePoint != null) {
                                addNewAnchor(edgePoint);
                            }
                        }
                        currentPath = dijkstra.getPathTo((int) imgX, (int) imgY);
                    }
                }
            }

            renderImage(BORDER);
            renderPath();
            renderSeedPoint();

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    private void cleanup() {

        if (window != NULL) {
            glfwFreeCallbacks(window);
            glfwDestroyWindow(window);
        }
        // 确保只执行一次terminate（新增）
        if (glfwInit()) {
            glfwTerminate(); //
        }

        if (texID != 0)
            glDeleteTextures(texID);
        glfwFreeCallbacks(window);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    private void setupCallbacks() {
        glfwSetKeyCallback(window, (window, key, scancode, action, mods) -> {
            if (action == GLFW_RELEASE) {
                if (key == GLFW_KEY_ESCAPE)
                    glfwSetWindowShouldClose(window, true);
                if (key == GLFW_KEY_0)
                    openImage();
            }
        });

        // 添加文件拖放回调
        glfwSetDropCallback(window, (window, count, names) -> {
            if (count > 0) {
                try (MemoryStack stack = stackPush()) {
                    // 获取拖放的文件路径
                    String path = MemoryUtil.memUTF8(names);
                    // 处理 macOS 的文件路径
                    if (System.getProperty("os.name").toLowerCase().contains("mac")) {
                        // 移除 file:// 前缀
                        if (path.startsWith("file://")) {
                            path = path.substring(7);
                        }
                        // 处理 URL 编码
                        try {
                            path = java.net.URLDecoder.decode(path, "UTF-8");
                        } catch (Exception e) {
                            System.err.println("Failed to decode path: " + e.getMessage());
                        }
                    }
                    // 移除路径中的引号
                    path = path.replace("\"", "");
                    // 移除可能的换行符
                    path = path.trim();

                    System.out.println("Loading image from: " + path);
                    File file = new File(path);
                    if (!file.exists()) {
                        System.err.println("File does not exist: " + path);
                        return;
                    }
                    if (!file.canRead()) {
                        System.err.println("Cannot read file: " + path);
                        return;
                    }

                    loadImageToTexture(path);
                    centerImage(BORDER);
                    isInteractionEnabled = true;
                } catch (Exception e) {
                    System.err.println("Failed to load image: " + e.getMessage());
                    e.printStackTrace();
                }
            }
        });

        glfwSetMouseButtonCallback(window, (window, button, action, mods) -> {
            if (!isInteractionEnabled)
                return;
            if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
                try (MemoryStack stack = stackPush()) {
                    DoubleBuffer xBuf = stack.mallocDouble(1);
                    DoubleBuffer yBuf = stack.mallocDouble(1);
                    glfwGetCursorPos(window, xBuf, yBuf);
                    double x = (xBuf.get(0) - panX) / scale;
                    double y = (yBuf.get(0) - panY) / scale;

                    if (isValidPoint((int) x, (int) y)) {
                        Point currentPoint = findMaxGradientPoint((int) x, (int) y, (int) adaptRadius);

                        if (seedPoint != null) {

                            if (distanceSq(currentPoint.x, currentPoint.y, seedPoint.x, seedPoint.y) >= SEARCH_RADIUS
                                    * SEARCH_RADIUS) {
                                return;
                            }

                            List<Point> path = dijkstra.getPathTo(currentPoint.x, currentPoint.y);
                            savedPaths.add(path);
                            savedSeedPoints.add(seedPoint);
                            seedPoint = currentPoint;
                            dijkstra = new DijkstraShortestPath(processor, SEARCH_RADIUS);
                            dijkstra.computeFrom(seedPoint.x, seedPoint.y);
                            // 闭环检测
                            if (firstSeedPoint != null && currentPoint.distance(firstSeedPoint) < adaptRadius) {
                                handleLoopClosure(firstSeedPoint, currentPoint);
                            }
                        } else {
                            seedPoint = currentPoint;
                            firstSeedPoint = seedPoint;
                            dijkstra = new DijkstraShortestPath(processor, SEARCH_RADIUS);
                            dijkstra.computeFrom(seedPoint.x, seedPoint.y);
                        }
                    }
                }
            }
        });

        glfwSetWindowSizeCallback(window, (window, width, height) -> {
            winWidth = width;
            winHeight = height;
            glViewport(0, 0, width, height);
            centerImage(BORDER);
        });
    }

    private Point findMaxGradientPoint(int x, int y, int radius) {
        int minX = Math.max(0, x - radius);
        int maxX = Math.min(imgWidth - 1, x + radius);
        int minY = Math.max(0, y - radius);
        int maxY = Math.min(imgHeight - 1, y + radius);

        double maxMagnitude = -1;
        int nx = x;
        int ny = y;

        for (int i = minX; i <= maxX; i++) {
            for (int j = minY; j <= maxY; j++) {
                double magnitude = processor.getGradient(i, j);
                if (magnitude > maxMagnitude) {
                    maxMagnitude = magnitude;
                    nx = i;
                    ny = j;
                }
            }
        }
        Point maxPoint = new Point(nx, ny);
        return maxPoint;
    }

    private Point projectToSearchRadius(Point current) {
        // 转换为相对锚点的坐标
        double dx = current.x - seedPoint.x;
        double dy = current.y - seedPoint.y;

        // 计算距离
        double distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < SEARCH_RADIUS) {
            return current; // 未超出范围直接返回
        }

        // 计算单位向量方向
        double ratio = (SEARCH_RADIUS * 1.05) / distance;
        return new Point(
                (int) (seedPoint.x + dx * ratio),
                (int) (seedPoint.y + dy * ratio));
    }

    private int distanceSq(int x1, int y1, int x2, int y2) {
        int dx = x1 - x2;
        int dy = y1 - y2;
        return dx * dx + dy * dy;
    }

    private Point findEdgePoint(Point current) {
        if (currentPath.isEmpty())
            return null;

        int targetIndex = (int) (currentPath.size() * cutPathRatio);
        targetIndex = Math.min(targetIndex, currentPath.size() - 1); // 防止越界

        return currentPath.get(targetIndex);
    }

    private List<Point> getPartialPath(List<Point> fullPath) {
        if (fullPath.isEmpty())
            return Collections.emptyList();

        // 计算保留点数（至少保留首节点）
        int keepCount = Math.max(1, (int) (fullPath.size() * cutPathRatio));
        return new ArrayList<>(fullPath.subList(0, keepCount));
    }

    private void addNewAnchor(Point newAnchor) {
        // 截取前80%路径
        List<Point> partialPath = getPartialPath(currentPath);

        // 仅当有效路径存在时才保存
        if (!partialPath.isEmpty()) {
            savedPaths.add(partialPath);
            savedSeedPoints.add(seedPoint);

            // 新锚点设为截取路径的终点（网页6中的路径终点更新策略）
            seedPoint = partialPath.get(partialPath.size() - 1);
        }

        // 重置Dijkstra计算起点（网页1的算法重置逻辑）
        dijkstra = new DijkstraShortestPath(processor, SEARCH_RADIUS);
        dijkstra.computeFrom(seedPoint.x, seedPoint.y);
    }

    private void createClippingWindow(List<Point> loopPath) {
        // 计算路径的边界框
        int minX = Integer.MAX_VALUE, maxX = Integer.MIN_VALUE;
        int minY = Integer.MAX_VALUE, maxY = Integer.MIN_VALUE;
        for (Point p : loopPath) {
            minX = Math.min(minX, p.x);
            maxX = Math.max(maxX, p.x);
            minY = Math.min(minY, p.y);
            maxY = Math.max(maxY, p.y);
        }

        // 计算原始尺寸
        int originalWidth = maxX - minX + 1;
        int originalHeight = maxY - minY + 1;

        // 计算缩放比例
        float widthRatio = 1.0f;
        float heightRatio = 1.0f;

        if (originalWidth > 2000) {
            widthRatio = 2000.0f / originalWidth;
        }
        if (originalHeight > 1125) {
            heightRatio = 1125.0f / originalHeight;
        }

        // 使用更严格的缩放比例（取两者中较小的）
        float scaleRatio = Math.min(widthRatio, heightRatio);

        // 如果不需要缩放，保持1.0
        if (scaleRatio >= 1.0f) {
            scaleRatio = 1.0f;
        }

        // 计算最终窗口尺寸
        int windowWidth = (int) (originalWidth * scaleRatio);
        int windowHeight = (int) (originalHeight * scaleRatio);

        // 创建共享上下文的新窗口
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
        long clipWindow = glfwCreateWindow(windowWidth, windowHeight, "Clipped View", NULL, window);
        glfwMakeContextCurrent(clipWindow);
        GL.createCapabilities();

        // 设置视口和投影
        glViewport(0, 0, windowWidth, windowHeight);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // 初始化模板和颜色缓冲
        glClearColor(0, 0, 0, 0);
        glClearStencil(0);
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        // ==== 步骤1: 绘制模板形状 ====
        glEnable(GL_STENCIL_TEST);
        glColorMask(false, false, false, false);
        glStencilFunc(GL_ALWAYS, 0, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);
        glStencilMask(0xFF);

        // 绘制闭合路径（应用缩放）
        glBegin(GL_POLYGON);
        for (Point p : loopPath) {
            float scaledX = (p.x - minX) * scaleRatio;
            float scaledY = (p.y - minY) * scaleRatio;
            glVertex2f(scaledX, scaledY);
        }
        glEnd();

        // ==== 步骤2: 绘制纹理内容 ====
        glColorMask(true, true, true, true);
        glStencilMask(0x00);
        glStencilFunc(GL_NOTEQUAL, 0, 0xFF);

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texID);

        // 计算纹理坐标（考虑缩放）
        float u1 = (float) minX / imgWidth;
        float u2 = (float) (maxX + 1) / imgWidth;
        float v1 = (float) minY / imgHeight;
        float v2 = (float) (maxY + 1) / imgHeight;

        // 绘制填充四边形（使用窗口尺寸）
        glBegin(GL_QUADS);
        glTexCoord2f(u1, v1);
        glVertex2f(0, 0);
        glTexCoord2f(u2, v1);
        glVertex2f(windowWidth, 0);
        glTexCoord2f(u2, v2);
        glVertex2f(windowWidth, windowHeight);
        glTexCoord2f(u1, v2);
        glVertex2f(0, windowHeight);
        glEnd();

        glfwSwapBuffers(clipWindow);

        // 事件循环
        while (!glfwWindowShouldClose(clipWindow)) {
            glfwPollEvents();
        }

        // 清理资源
        glDisable(GL_STENCIL_TEST);
        glfwDestroyWindow(clipWindow);
        GL.setCapabilities(null);
    }

    private void handleLoopClosure(Point loopStart, Point currentPoint) {
        // 收集闭环路径
        int startIndex = savedSeedPoints.indexOf(loopStart);
        if (startIndex == -1)
            return;

        List<List<Point>> closurePaths = new ArrayList<>(savedPaths.subList(startIndex, savedPaths.size()));
        List<Point> fullLoop = new ArrayList<>();
        for (List<Point> path : closurePaths) {
            fullLoop.addAll(path);
        }

        // 计算最后一段路径
        DijkstraShortestPath finalDijkstra = new DijkstraShortestPath(processor, SEARCH_RADIUS);
        finalDijkstra.computeFrom(currentPoint.x, currentPoint.y);
        List<Point> lastSegment = finalDijkstra.getPathTo(loopStart.x, loopStart.y);
        fullLoop.addAll(lastSegment);

        createClippingWindow(fullLoop);
    }

    private boolean isValidPoint(int x, int y) {
        return x >= 0 && x < imgWidth && y >= 0 && y < imgHeight;
    }

    private void loadDefaultImage() {
        String defaultPath = "desk.png"; // 根目录下的默认图片
        if (new File(defaultPath).exists()) {
            loadImageToTexture(defaultPath);
        } else {
            System.err.println("Default image not found: " + defaultPath);
        }
        isInteractionEnabled = false;
    }

    private void openImage() {
        try {
            // 使用更简单的 AppleScript 命令
            ProcessBuilder pb = new ProcessBuilder("osascript", "-e",
                    "POSIX path of (choose file of type {\"public.image\"} with prompt \"选择图片\")");
            Process p = pb.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String path = reader.readLine();

            if (path != null && !path.isEmpty()) {
                // 移除可能的换行符
                path = path.trim();

                System.out.println("Selected image: " + path);
                File file = new File(path);
                if (!file.exists()) {
                    System.err.println("File does not exist: " + path);
                    return;
                }
                if (!file.canRead()) {
                    System.err.println("Cannot read file: " + path);
                    return;
                }

                loadImageToTexture(path);
                centerImage(BORDER);
                isInteractionEnabled = true;
            }
        } catch (Exception e) {
            System.err.println("Failed to open file dialog: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void loadImageToTexture(String path) {
        if (texID != 0) {
            glDeleteTextures(texID);
            texID = 0;
        }
        try (MemoryStack stack = stackPush()) {
            IntBuffer w = stack.mallocInt(1);
            IntBuffer h = stack.mallocInt(1);
            IntBuffer comp = stack.mallocInt(1);

            System.out.println("Attempting to load image: " + path);
            File file = new File(path);
            if (!file.exists()) {
                throw new IOException("File does not exist: " + path);
            }
            if (!file.canRead()) {
                throw new IOException("Cannot read file: " + path);
            }

            ByteBuffer data = stbi_load(path, w, h, comp, 3);
            if (data == null) {
                String error = stbi_failure_reason();
                throw new IOException("Image load failed: " + error);
            }

            imgWidth = w.get(0);
            imgHeight = h.get(0);
            System.out.println("Image loaded successfully. Dimensions: " + imgWidth + "x" + imgHeight);

            texID = glGenTextures();
            glBindTexture(GL_TEXTURE_2D, texID);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imgWidth, imgHeight, 0,
                    GL_RGB, GL_UNSIGNED_BYTE, data);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            stbi_image_free(data);

            ByteBuffer grayData = loadGrayImage(path);
            processor = new ImageProcessor(grayData, imgWidth, imgHeight);
            adaptRadius = Math.sqrt(processor.getHeight() * processor.getWidth()) / 150.0;

        } catch (Exception e) {
            System.err.println("Error loading image: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("Failed to load image", e);
        }
    }

    // 修改后的centerImage()方法
    private void centerImage(int BORDER) {
        if (imgWidth == 0 || imgHeight == 0)
            return;

        try (MemoryStack stack = stackPush()) {
            IntBuffer w = stack.mallocInt(1);
            IntBuffer h = stack.mallocInt(1);
            glfwGetWindowSize(window, w, h);

            // 计算可用空间（窗口尺寸减去双倍边框）
            int availableWidth = w.get(0) - 2 * BORDER;
            int availableHeight = h.get(0) - 2 * BORDER;

            // 防止负值
            availableWidth = Math.max(availableWidth, 10);
            availableHeight = Math.max(availableHeight, 10);

            // 计算带边框的缩放比例
            float scaleX = (float) availableWidth / imgWidth;
            float scaleY = (float) availableHeight / imgHeight;
            scale = Math.min(scaleX, scaleY);

            // 计算带边框的居中位置
            panX = BORDER + (availableWidth - imgWidth * scale) / 2;
            panY = BORDER + (availableHeight - imgHeight * scale) / 2;
        }
    }

    private ByteBuffer loadGrayImage(String path) {
        try (MemoryStack stack = stackPush()) {
            IntBuffer w = stack.mallocInt(1);
            IntBuffer h = stack.mallocInt(1);
            IntBuffer comp = stack.mallocInt(1);
            ByteBuffer data = stbi_load(path, w, h, comp, 1);
            if (data == null)
                throw new IOException("Gray image load failed");
            return data;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void renderImage(int BORDER) {
        if (texID == 0)
            return;

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, winWidth, winHeight, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glOrtho(-BORDER, winWidth + BORDER, winHeight + BORDER, -BORDER, -1, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glBindTexture(GL_TEXTURE_2D, texID);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(panX, panY);
        glTexCoord2f(1, 0);
        glVertex2f(panX + imgWidth * scale, panY);
        glTexCoord2f(1, 1);
        glVertex2f(panX + imgWidth * scale, panY + imgHeight * scale);
        glTexCoord2f(0, 1);
        glVertex2f(panX, panY + imgHeight * scale);
        glEnd();
    }

    private void renderPath() {
        // 绘制历史路径
        for (List<Point> path : savedPaths) {
            if (path.isEmpty())
                continue;

            glDisable(GL_TEXTURE_2D);
            glColor3f(1, 0, 0); // 历史路径为红色
            glLineWidth(2);
            glBegin(GL_LINE_STRIP);
            for (Point p : path) {
                glVertex2f(panX + p.x * scale, panY + p.y * scale);
            }
            glEnd();
            glEnable(GL_TEXTURE_2D);
        }

        // 绘制当前路径
        if (currentPath.isEmpty())
            return;

        glDisable(GL_TEXTURE_2D);
        glColor3f(0, 0, 1); // 当前路径为蓝色以区分
        glLineWidth(2);
        glBegin(GL_LINE_STRIP);
        for (Point p : currentPath) {
            glVertex2f(panX + p.x * scale, panY + p.y * scale);
        }
        glEnd();
        glEnable(GL_TEXTURE_2D);
        glColor3f(1, 1, 1);
    }

    private void renderSeedPoint() {
        // 绘制历史起点
        for (Point p : savedSeedPoints) {
            glDisable(GL_TEXTURE_2D);
            glColor3f(0, 1, 0); // 绿色
            glPointSize(8);
            glBegin(GL_POINTS);
            glVertex2f(panX + p.x * scale, panY + p.y * scale);
            glEnd();
            glEnable(GL_TEXTURE_2D);
        }

        // 绘制当前起点
        if (seedPoint == null)
            return;

        glDisable(GL_TEXTURE_2D);
        glColor3f(1, 0, 1); // 洋红色以区分
        glPointSize(8);
        glBegin(GL_POINTS);
        glVertex2f(panX + seedPoint.x * scale, panY + seedPoint.y * scale);
        glEnd();
        glEnable(GL_TEXTURE_2D);
        glColor3f(1, 1, 1);
    }

}