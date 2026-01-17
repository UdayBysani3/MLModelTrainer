"use client";
import React from "react";
import { cn } from "../../lib/utils";

export const BackgroundBeams = ({ className }) => {
    return (
        <div
            className={cn(
                "fixed inset-0 z-0 flex flex-col items-center justify-center bg-neutral-950",
                className
            )}
        >
            <div className="absolute inset-0 bg-neutral-950 [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]" />
            <div className="absolute inset-0 bg-fixed bg-center [mask-image:linear-gradient(to_bottom,transparent,black)] pointer-events-none">
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-neutral-500/10 to-transparent h-full w-full [transform:skewY(-20deg)] animate-pulse" />
            </div>
            {/* Grid Pattern */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none" />
        </div>
    );
};
