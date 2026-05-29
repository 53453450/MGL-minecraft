package mglprobe;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.Instrumentation;
import java.security.ProtectionDomain;
import java.util.jar.JarFile;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

public final class TexProbeAgent {
    public static void premain(String args, Instrumentation inst) {
        install("premain", args, inst, false);
    }

    public static void agentmain(String args, Instrumentation inst) {
        install("agentmain", args, inst, true);
    }

    private static synchronized void install(String phase, String args, Instrumentation inst, boolean retransformLoaded) {
        applyArgs(args);
        exposeRuntimeToBootstrap(inst);
        log("MGLJ TEXPROBE agent loaded args=" + args
                + " phase=" + phase
                + " command=" + System.getProperty("sun.java.command", "-"));
        Thread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
            @Override
            public void uncaughtException(Thread thread, Throwable error) {
                log("MGLJ TEXPROBE uncaught thread=" + thread.getName() + " error=" + error, error);
            }
        });
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                log("MGLJ TEXPROBE shutdown");
            }
        }, "mgl-tex-probe-shutdown"));
        inst.addTransformer(new Transformer(), false);
        if (retransformLoaded && inst.isRetransformClassesSupported()) {
            retransformIfLoaded(inst);
        }
    }

    private static void exposeRuntimeToBootstrap(Instrumentation inst) {
        String path = System.getProperty("mgl.texprobe.agentJar",
                System.getProperty("mgl.texprobe.agent"));
        if (path == null || path.isEmpty()) {
            return;
        }
        try {
            inst.appendToBootstrapClassLoaderSearch(new JarFile(path));
            log("MGLJ TEXPROBE bootstrap append " + path);
        } catch (Throwable t) {
            log("MGLJ TEXPROBE bootstrap append failed path=" + path + " error=" + t, t);
        }
    }

    private static void applyArgs(String args) {
        if (args == null || args.trim().isEmpty()) {
            return;
        }
        String[] entries = args.split("[,;]");
        for (String entry : entries) {
            int eq = entry.indexOf('=');
            if (eq <= 0) {
                continue;
            }
            String key = entry.substring(0, eq).trim();
            String value = entry.substring(eq + 1).trim();
            if (key.isEmpty()) {
                continue;
            }
            if (!key.startsWith("mgl.texprobe.")) {
                key = "mgl.texprobe." + key;
            }
            System.setProperty(key, value);
        }
    }

    private static void retransformIfLoaded(Instrumentation inst) {
        for (Class<?> klass : inst.getAllLoadedClasses()) {
            String name = klass.getName();
            if (!"org.lwjgl.opengl.GL11C".equals(name)
                    && !"org.lwjgl.opengl.GL12C".equals(name)
                    && !"org.lwjgl.opengl.GL45C".equals(name)) {
                continue;
            }
            try {
                if (inst.isModifiableClass(klass)) {
                    log("MGLJ TEXPROBE retransform " + name);
                    inst.retransformClasses(klass);
                } else {
                    log("MGLJ TEXPROBE cannot retransform " + name + " modifiable=false");
                }
            } catch (Throwable t) {
                log("MGLJ TEXPROBE retransform failed class=" + name + " error=" + t, t);
            }
        }
    }

    private static synchronized void log(String message) {
        String path = System.getProperty("mgl.texprobe.log",
                System.getProperty("user.home") + "/Documents/java-tex-probe.log");
        try {
            PrintStream out = new PrintStream(new FileOutputStream(path, true), true, "UTF-8");
            try {
                out.println(message);
            } finally {
                out.close();
            }
        } catch (Throwable ignored) {
            // Keep the launcher protocol clean even if the side log cannot be opened.
        }
    }

    private static synchronized void log(String message, Throwable error) {
        String path = System.getProperty("mgl.texprobe.log",
                System.getProperty("user.home") + "/Documents/java-tex-probe.log");
        try {
            PrintStream out = new PrintStream(new FileOutputStream(path, true), true, "UTF-8");
            try {
                out.println(message);
                error.printStackTrace(out);
            } finally {
                out.close();
            }
        } catch (Throwable ignored) {
            // Keep the launcher protocol clean even if the side log cannot be opened.
        }
    }

    static final class Transformer implements ClassFileTransformer {
        @Override
        public byte[] transform(ClassLoader loader, String className, Class<?> classBeingRedefined,
                                ProtectionDomain protectionDomain, byte[] classfileBuffer) {
            if (!"org/lwjgl/opengl/GL11C".equals(className)
                    && !"org/lwjgl/opengl/GL12C".equals(className)
                    && !"org/lwjgl/opengl/GL45C".equals(className)) {
                return null;
            }

            try {
                ClassReader cr = new ClassReader(classfileBuffer);
                ClassWriter cw = new ClassWriter(cr, ClassWriter.COMPUTE_MAXS);
                ClassVisitor cv = new ClassVisitor(Opcodes.ASM7, cw) {
                    @Override
                    public MethodVisitor visitMethod(int access, String name, String desc, String signature, String[] exceptions) {
                        MethodVisitor mv = super.visitMethod(access, name, desc, signature, exceptions);
                        if ("glTexSubImage2D".equals(name)) {
                            if ("(IIIIIIIILjava/nio/ByteBuffer;)V".equals(desc)
                                    || "(IIIIIIIILjava/nio/ShortBuffer;)V".equals(desc)
                                    || "(IIIIIIIILjava/nio/IntBuffer;)V".equals(desc)
                                    || "(IIIIIIIILjava/nio/FloatBuffer;)V".equals(desc)
                                    || "(IIIIIIIILjava/nio/DoubleBuffer;)V".equals(desc)) {
                                return new SubImageBufferProbe(mv, className, desc);
                            }
                            if ("(IIIIIIIIJ)V".equals(desc)) {
                                return new SubImageLongProbe(mv, className, desc);
                            }
                        }
                        if ("glTexImage2D".equals(name)) {
                            if ("(IIIIIIIILjava/nio/ByteBuffer;)V".equals(desc)
                                    || "(IIIIIIIILjava/nio/ShortBuffer;)V".equals(desc)
                                    || "(IIIIIIIILjava/nio/IntBuffer;)V".equals(desc)
                                    || "(IIIIIIIILjava/nio/FloatBuffer;)V".equals(desc)
                                    || "(IIIIIIIILjava/nio/DoubleBuffer;)V".equals(desc)) {
                                return new TexImageBufferProbe(mv, className, desc);
                            }
                            if ("(IIIIIIIIJ)V".equals(desc)) {
                                return new TexImageLongProbe(mv, className, desc);
                            }
                        }
                        if ("glTexParameteri".equals(name) && "(III)V".equals(desc)) {
                            return new TexParamProbe(mv, className, "i");
                        }
                        if ("glTexParameterf".equals(name) && "(IIF)V".equals(desc)) {
                            return new TexParamProbe(mv, className, "f");
                        }
                        return mv;
                    }
                };
                cr.accept(cv, 0);
                log("MGLJ TEXPROBE transformed " + className);
                return cw.toByteArray();
            } catch (Throwable t) {
                log("MGLJ TEXPROBE transform failed class=" + className + " error=" + t, t);
                return null;
            }
        }
    }

    // --- glTexSubImage2D probes ---

    static final class SubImageBufferProbe extends MethodVisitor implements Opcodes {
        private final String owner, desc;
        SubImageBufferProbe(MethodVisitor mv, String owner, String desc) {
            super(ASM7, mv);
            this.owner = owner; this.desc = desc;
        }
        @Override
        public void visitCode() {
            super.visitCode();
            mv.visitLdcInsn(owner.replace('/', '.'));
            mv.visitLdcInsn(desc);
            for (int i = 0; i < 8; i++) mv.visitVarInsn(ILOAD, i);
            mv.visitVarInsn(ALOAD, 8);
            mv.visitMethodInsn(INVOKESTATIC, "mglprobe/TexProbeRuntime", "logTexSubImage2DBuffer",
                    "(Ljava/lang/String;Ljava/lang/String;IIIIIIIILjava/nio/Buffer;)V", false);
        }
    }

    static final class SubImageLongProbe extends MethodVisitor implements Opcodes {
        private final String owner, desc;
        SubImageLongProbe(MethodVisitor mv, String owner, String desc) {
            super(ASM7, mv);
            this.owner = owner; this.desc = desc;
        }
        @Override
        public void visitCode() {
            super.visitCode();
            mv.visitLdcInsn(owner.replace('/', '.'));
            mv.visitLdcInsn(desc);
            for (int i = 0; i < 8; i++) mv.visitVarInsn(ILOAD, i);
            mv.visitVarInsn(LLOAD, 8);
            mv.visitMethodInsn(INVOKESTATIC, "mglprobe/TexProbeRuntime", "logTexSubImage2DAddress",
                    "(Ljava/lang/String;Ljava/lang/String;IIIIIIIIJ)V", false);
        }
    }

    // --- glTexImage2D probes ---

    static final class TexImageBufferProbe extends MethodVisitor implements Opcodes {
        private final String owner, desc;
        TexImageBufferProbe(MethodVisitor mv, String owner, String desc) {
            super(ASM7, mv);
            this.owner = owner; this.desc = desc;
        }
        @Override
        public void visitCode() {
            super.visitCode();
            mv.visitLdcInsn(owner.replace('/', '.'));
            mv.visitLdcInsn(desc);
            for (int i = 0; i < 8; i++) mv.visitVarInsn(ILOAD, i);
            mv.visitVarInsn(ALOAD, 8); // pixels is arg index 8 (target,level,internal,w,h,border,fmt,type,pixels)
            mv.visitMethodInsn(INVOKESTATIC, "mglprobe/TexProbeRuntime", "logTexImage2DBuffer",
                    "(Ljava/lang/String;Ljava/lang/String;IIIIIIIILjava/nio/Buffer;)V", false);
        }
    }

    static final class TexImageLongProbe extends MethodVisitor implements Opcodes {
        private final String owner, desc;
        TexImageLongProbe(MethodVisitor mv, String owner, String desc) {
            super(ASM7, mv);
            this.owner = owner; this.desc = desc;
        }
        @Override
        public void visitCode() {
            super.visitCode();
            mv.visitLdcInsn(owner.replace('/', '.'));
            mv.visitLdcInsn(desc);
            for (int i = 0; i < 8; i++) mv.visitVarInsn(ILOAD, i);
            mv.visitVarInsn(LLOAD, 8); // pixels is arg index 8 (long takes slots 8+9)
            mv.visitMethodInsn(INVOKESTATIC, "mglprobe/TexProbeRuntime", "logTexImage2DAddress",
                    "(Ljava/lang/String;Ljava/lang/String;IIIIIIIIJ)V", false);
        }
    }

    // --- glTexParameter probes ---

    static final class TexParamProbe extends MethodVisitor implements Opcodes {
        private final String owner, variant;
        TexParamProbe(MethodVisitor mv, String owner, String variant) {
            super(ASM7, mv);
            this.owner = owner; this.variant = variant;
        }
        @Override
        public void visitCode() {
            super.visitCode();
            mv.visitLdcInsn(owner.replace('/', '.'));
            mv.visitLdcInsn(variant);
            mv.visitVarInsn(ILOAD, 0); // target
            mv.visitVarInsn(ILOAD, 1); // pname
            if ("f".equals(variant)) {
                mv.visitVarInsn(FLOAD, 2); // param (float)
                mv.visitMethodInsn(INVOKESTATIC, "mglprobe/TexProbeRuntime", "logTexParameterF",
                        "(Ljava/lang/String;Ljava/lang/String;IIF)V", false);
            } else {
                mv.visitVarInsn(ILOAD, 2); // param (int)
                mv.visitMethodInsn(INVOKESTATIC, "mglprobe/TexProbeRuntime", "logTexParameter",
                        "(Ljava/lang/String;Ljava/lang/String;III)V", false);
            }
        }
    }
}
