"use client";
import { cn } from "../../lib/utils";
import { motion } from "framer-motion";

export const TypewriterEffect = ({
    words,
    className,
    cursorClassName,
}) => {
    const wordsArray = words.map((word) => {
        return {
            ...word,
            text: word.text.split(""),
        };
    });

    const renderWords = () => {
        return (
            <div className="inline relative">
                {wordsArray.map((word, idx) => {
                    return (
                        <div key={`word-${idx}`} className="inline-block mr-1.5 md:mr-2">
                            {word.text.map((char, index) => (
                                <motion.span
                                    initial={{
                                        opacity: 0,
                                    }}
                                    animate={{
                                        opacity: 1,
                                    }}
                                    transition={{
                                        duration: 0.05,
                                        delay: idx * 0.1 + index * 0.03,
                                        ease: "easeInOut",
                                    }}
                                    key={`char-${index}`}
                                    className={cn(`dark:text-white text-black`, word.className)}
                                >
                                    {char}
                                </motion.span>
                            ))}
                        </div>
                    );
                })}
            </div>
        );
    };
    return (
        <div
            className={cn(
                "text-base sm:text-lg md:text-xl lg:text-2xl font-bold text-start flex items-center",
                className
            )}
        >
            {renderWords()}
            <motion.span
                initial={{
                    opacity: 0,
                }}
                animate={{
                    opacity: 1,
                }}
                transition={{
                    duration: 0.8,
                    repeat: Infinity,
                    repeatType: "reverse",
                }}
                className={cn(
                    "inline-block rounded-sm w-[4px] h-4 md:h-6 lg:h-8 bg-blue-500 cursor-blinking",
                    cursorClassName
                )}
            ></motion.span>
        </div>
    );
};
