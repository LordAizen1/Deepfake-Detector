"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";

interface Props {
  onMediaSelected: (file: File, preview: string) => void;
  disabled?: boolean;
}

export default function ImageUpload({ onMediaSelected, disabled }: Props) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;
      onMediaSelected(file, URL.createObjectURL(file));
    },
    [onMediaSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpg", ".jpeg", ".png", ".webp"],
      "video/*": [".mp4", ".webm", ".mov"],
    },
    maxFiles: 1,
    disabled,
  });

  return (
    <div
      {...getRootProps()}
      className={`w-full h-full flex flex-col items-center justify-center cursor-pointer transition-colors duration-300 ${
        isDragActive ? "bg-surface-container-high/40" : ""
      } ${disabled ? "pointer-events-none opacity-50" : ""}`}
    >
      <input {...getInputProps()} />
      <span className="material-symbols-outlined text-outline text-5xl mb-4">
        upload_file
      </span>
      <p className="font-label text-xs uppercase tracking-[0.1em] text-on-surface-variant">
        {isDragActive ? "Drop to analyze" : "Click or drag an image or video"}
      </p>
      <p className="font-label text-[10px] uppercase tracking-widest text-outline-variant mt-1">
        JPG · PNG · WEBP · MP4 · MOV · max 200MB
      </p>
    </div>
  );
}
