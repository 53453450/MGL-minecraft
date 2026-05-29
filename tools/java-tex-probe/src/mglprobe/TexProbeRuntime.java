package mglprobe;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.nio.Buffer;
import java.nio.ByteBuffer;

public final class TexProbeRuntime {
    private static final boolean STACK = !"0".equals(System.getProperty("mgl.texprobe.stack", "1"));
    private static final int MAX_LOG = Integer.getInteger("mgl.texprobe.max", 80);
    private static final int TARGET_W = Integer.getInteger("mgl.texprobe.width", 512);
    private static final int TARGET_H = Integer.getInteger("mgl.texprobe.height", 512);
    private static final boolean LOG_PARAMS = !"0".equals(System.getProperty("mgl.texprobe.params", "1"));
    private static final PrintStream OUT = openLog();
    private static int seen;

    private TexProbeRuntime() {}

    private static boolean shouldLog(int width, int height) {
        return TARGET_W <= 0 || TARGET_H <= 0 || (width == TARGET_W && height == TARGET_H);
    }

    private static synchronized int nextId() {
        if (seen >= MAX_LOG) return -1;
        return ++seen;
    }

    // --- glTexImage2D ---

    public static void logTexImage2DBuffer(String owner, String desc,
            int target, int level, int internalformat, int width, int height, int border, int format, int type,
            Buffer pixels) {
        if (!shouldLog(width, height)) return;
        int id = nextId();
        if (id < 0) return;
        long addr = memAddressSafe(pixels);
        long hash = hashBufferHead(pixels, 256);
        boolean zero = looksZero(pixels, 256);
        String dump = dumpBufferHead(pixels, 64);
        GLTextureState state = readTextureState();
        OUT.printf(
                "MGLJ TEXIMAGE #%d %s target=0x%x activeUnit=%d bound2D=%d level=%d internal=0x%x size=%dx%d border=%d format=0x%x type=0x%x addr=0x%x hash256=0x%016x zero256=%s head=%s%n",
                id, owner, target, state.activeUnit, state.bound2D, level, internalformat, width, height, border, format, type, addr, hash, zero, dump);
        logStack(id);
    }

    public static void logTexImage2DAddress(String owner, String desc,
            int target, int level, int internalformat, int width, int height, int border, int format, int type,
            long pixels) {
        if (!shouldLog(width, height)) return;
        int id = nextId();
        if (id < 0) return;
        GLTextureState state = readTextureState();
        OUT.printf(
                "MGLJ TEXIMAGE #%d %s target=0x%x activeUnit=%d bound2D=%d level=%d internal=0x%x size=%dx%d border=%d format=0x%x type=0x%x pboOffset=0x%x%n",
                id, owner, target, state.activeUnit, state.bound2D, level, internalformat, width, height, border, format, type, pixels);
        logStack(id);
    }

    // --- glTexSubImage2D (existing) ---

    public static void logTexSubImage2DBuffer(String owner, String desc,
            int target, int level, int xoffset, int yoffset, int width, int height, int format, int type,
            Buffer pixels) {
        if (!shouldLog(width, height)) return;
        int id = nextId();
        if (id < 0) return;
        long address = memAddressSafe(pixels);
        long hash = hashBufferHead(pixels, 256);
        boolean zero = looksZero(pixels, 256);
        String dump = dumpBufferHead(pixels, 32);
        GLTextureState state = readTextureState();
        OUT.printf(
                "MGLJ TEXPROBE #%d %s%s target=0x%x activeUnit=%d bound2D=%d level=%d off=(%d,%d) size=%dx%d format=0x%x type=0x%x addr=0x%x hash256=0x%016x zero256=%s head=%s%n",
                id, owner, desc, target, state.activeUnit, state.bound2D, level, xoffset, yoffset, width, height, format, type, address, hash, zero, dump);
        logStack(id);
    }

    public static void logTexSubImage2DAddress(String owner, String desc,
            int target, int level, int xoffset, int yoffset, int width, int height, int format, int type,
            long pixels) {
        if (!shouldLog(width, height)) return;
        int id = nextId();
        if (id < 0) return;
        GLTextureState state = readTextureState();
        OUT.printf(
                "MGLJ TEXPROBE #%d %s%s target=0x%x activeUnit=%d bound2D=%d level=%d off=(%d,%d) size=%dx%d format=0x%x type=0x%x rawAddressOrPboOffset=0x%x%n",
                id, owner, desc, target, state.activeUnit, state.bound2D, level, xoffset, yoffset, width, height, format, type, pixels);
        logStack(id);
    }

    // --- glTexParameter ---

    public static void logTexParameter(String owner, String variant, int target, int pname, int param) {
        if (!LOG_PARAMS) return;
        int id = nextId();
        if (id < 0) return;
        OUT.printf(
                "MGLJ TEXPARAM #%d %s target=0x%x pname=0x%x param=0x%x (%d)%n",
                id, owner, target, pname, param, param);
    }

    public static void logTexParameterF(String owner, String variant, int target, int pname, float param) {
        if (!LOG_PARAMS) return;
        int id = nextId();
        if (id < 0) return;
        OUT.printf(
                "MGLJ TEXPARAM #%d %s target=0x%x pname=0x%x param=%f%n",
                id, owner, target, pname, param);
    }

    // --- helpers ---

    private static PrintStream openLog() {
        String path = System.getProperty("mgl.texprobe.log",
                System.getProperty("user.home") + "/Documents/java-tex-probe.log");
        try {
            return new PrintStream(new FileOutputStream(path, true), true, "UTF-8");
        } catch (Throwable ignored) {
            return System.err;
        }
    }

    private static void logStack(int id) {
        if (!STACK) return;
        StackTraceElement[] stack = Thread.currentThread().getStackTrace();
        OUT.println("MGLJ TEXPROBE #" + id + " stack:");
        for (int i = 2; i < Math.min(stack.length, 28); i++) {
            String s = stack[i].toString();
            if (s.startsWith("org.lwjgl.opengl.GL11C.") ||
                s.startsWith("org.lwjgl.opengl.GL12C.") ||
                s.startsWith("org.lwjgl.opengl.GL45C.") ||
                s.startsWith("com.mojang.") ||
                s.startsWith("net.minecraft.") ||
                s.matches("^[a-z]{2,4}\\..*|^[a-z]{2,4}\\(.*")) {
                OUT.println("  at " + s);
            }
        }
        OUT.flush();
    }

    private static long memAddressSafe(Buffer b) {
        if (b == null) return 0L;
        try {
            Class<?> mu = classForName("org.lwjgl.system.MemoryUtil");
            Method m = mu.getMethod("memAddressSafe", Buffer.class);
            Object v = m.invoke(null, b);
            return ((Long)v).longValue();
        } catch (Throwable ignored) {
            return 0L;
        }
    }

    private static GLTextureState readTextureState() {
        GLTextureState state = new GLTextureState();
        try {
            Class<?> gl11 = classForName("org.lwjgl.opengl.GL11C");
            Method getInteger = gl11.getMethod("glGetInteger", int.class);
            int active = ((Integer)getInteger.invoke(null, 0x84e0)).intValue(); // GL_ACTIVE_TEXTURE
            int bound2D = ((Integer)getInteger.invoke(null, 0x8069)).intValue(); // GL_TEXTURE_BINDING_2D
            state.activeUnit = active >= 0x84c0 ? active - 0x84c0 : active;
            state.bound2D = bound2D;
        } catch (Throwable ignored) {
            state.activeUnit = -1;
            state.bound2D = -1;
        }
        return state;
    }

    private static Class<?> classForName(String name) throws ClassNotFoundException {
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        if (loader != null) {
            try {
                return Class.forName(name, false, loader);
            } catch (ClassNotFoundException ignored) {
                // Fall through to the defining loader.
            }
        }
        return Class.forName(name);
    }

    private static final class GLTextureState {
        int activeUnit = -1;
        int bound2D = -1;
    }

    private static String dumpBufferHead(Buffer b, int maxBytes) {
        if (!(b instanceof ByteBuffer)) return b == null ? "-" : "non-byte-buffer";
        ByteBuffer bb = ((ByteBuffer)b).duplicate();
        int n = Math.min(Math.min(maxBytes, bb.remaining()), 64);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            if (i > 0) sb.append(':');
            int v = bb.get(bb.position() + i) & 0xff;
            if (v < 16) sb.append('0');
            sb.append(Integer.toHexString(v));
        }
        return sb.toString();
    }

    private static long hashBufferHead(Buffer b, int maxBytes) {
        if (!(b instanceof ByteBuffer)) return 0L;
        ByteBuffer bb = ((ByteBuffer)b).duplicate();
        int n = Math.min(maxBytes, bb.remaining());
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < n; i++) {
            h ^= (bb.get(bb.position() + i) & 0xff);
            h *= 0x100000001b3L;
        }
        return h;
    }

    private static boolean looksZero(Buffer b, int maxBytes) {
        if (!(b instanceof ByteBuffer)) return false;
        ByteBuffer bb = ((ByteBuffer)b).duplicate();
        int n = Math.min(maxBytes, bb.remaining());
        if (n <= 0) return false;
        for (int i = 0; i < n; i++) {
            if (bb.get(bb.position() + i) != 0) return false;
        }
        return true;
    }
}
