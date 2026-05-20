"use client";

import Image from "next/image";
import { useCallback, useState } from "react";

interface ImageUploadProps {
  onImageSelect: (file: File, previewUrl: string) => void;
}

export default function ImageUpload({ onImageSelect }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (!file || !file.type.match(/^image\/(jpeg|jpg|png)$/)) return;
      const url = URL.createObjectURL(file);
      setPreview(url);
      onImageSelect(file, url);
    },
    [onImageSelect]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-slate-300">Coral Image</label>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={`relative flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition ${
          dragOver
            ? "border-reef-teal bg-reef-teal/10"
            : "border-slate-600 bg-ocean-800/50 hover:border-reef-teal/50"
        }`}
      >
        <input
          type="file"
          accept="image/jpeg,image/png"
          className="absolute inset-0 cursor-pointer opacity-0"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        {preview ? (
          <div className="relative h-48 w-full p-2">
            <Image
              src={preview}
              alt="Coral preview"
              fill
              className="rounded-lg object-contain"
              unoptimized
            />
          </div>
        ) : (
          <div className="p-6 text-center text-slate-400">
            <p className="text-4xl">📷</p>
            <p className="mt-2 text-sm">Drag & drop or click to upload</p>
            <p className="text-xs text-slate-500">JPG or PNG</p>
          </div>
        )}
      </div>
    </div>
  );
}
