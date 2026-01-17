import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { cn } from "../../lib/utils";

export const Modal = ({ isOpen, onClose, children, className }) => {
    return (
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center overflow-auto bg-black/80 backdrop-blur-md p-4">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 20 }}
                        transition={{ type: "spring", duration: 0.5 }}
                        className={cn(
                            "relative w-full max-w-5xl rounded-2xl bg-gradient-to-br from-neutral-900 to-black border border-white/10 shadow-2xl overflow-hidden",
                            className
                        )}
                    >
                        <button
                            onClick={onClose}
                            className="absolute right-4 top-4 z-10 p-2 rounded-full bg-white/5 hover:bg-white/10 transition-all duration-200 group"
                        >
                            <X className="w-5 h-5 text-neutral-400 group-hover:text-white transition-colors" />
                        </button>
                        <div className="p-6 md:p-8 max-h-[85vh] overflow-y-auto custom-scrollbar">
                            {children}
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    );
};
