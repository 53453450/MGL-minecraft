package mglprobe;

import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.Instrumentation;
import java.security.ProtectionDomain;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

public final class TexProbeAgent {
    public static void premain(String args, Instrumentation inst) {
        System.err.println("MGLJ TEXPROBE agent loaded args=" + args);
        inst.addTransformer(new Transformer(), false);
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
                System.err.println("MGLJ TEXPROBE transformed " + className);
                return cw.toByteArray();
            } catch (Throwable t) {
                System.err.println("MGLJ TEXPROBE transform failed class=" + className + " error=" + t);
                t.printStackTrace(System.err);
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
