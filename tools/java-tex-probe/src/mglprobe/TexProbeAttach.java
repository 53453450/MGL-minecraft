package mglprobe;

import com.sun.tools.attach.VirtualMachine;

public final class TexProbeAttach {
    private TexProbeAttach() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 3) {
            System.err.println("usage: TexProbeAttach <pid> <agent-jar> [agent-args]");
            System.exit(2);
        }

        VirtualMachine vm = VirtualMachine.attach(args[0]);
        try {
            vm.loadAgent(args[1], args.length == 3 ? args[2] : "");
        } finally {
            vm.detach();
        }
    }
}
