import React, { useState, useEffect } from "react";
import { Card, CardContent } from "../ui/card";
import { Toaster } from "sonner";
import { FileText } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

function FilePanel() {
  const [open, setOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [files, setFiles] = useState([]);
  const [previewData, setPreviewData] = useState(null);
  const [loadingPreview, setLoadingPreview] = useState(false);

  const fetchFiles = async () => {
    try {
      const res = await fetch("http://localhost:8000/file/binance", {
        credentials: "include", 
      });
      if (!res.ok) throw new Error("Failed to fetch files");
      const data = await res.json();
      setFiles(data);
    } catch (err) {
      console.error("Error fetching files:", err);
    }
  };

  const fetchFilePreview = async (filename) => {
    try {
      setLoadingPreview(true);
      const res = await fetch(
        `http://localhost:8000/file/binance/${filename}`,
        { credentials: "include" }
      );
      if (!res.ok) throw new Error("Failed to fetch preview");
      const data = await res.json();
      setPreviewData(data);
    } catch (err) {
      console.error("Error fetching preview:", err);
    } finally {
      setLoadingPreview(false);
    }
  };

  const handleDoubleClick = (file) => {
    setSelectedFile(file);
    setOpen(true);
    fetchFilePreview(file.name);
  };

  useEffect(() => {
    fetchFiles();
    const interval = setInterval(fetchFiles, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full gap-6">
      <Toaster position="top-right" />
      <Card className="shadow-lg rounded-2xl h-[52vh] py-4 flex flex-col">
        <CardContent className="space-y-2 px-6 py-2 flex-1 flex flex-col h-full">
          <div className="space-y-1">
            <h2 className="text-xl font-bold tracking-tight">
              Files from Binance
            </h2>
            <p className="text-sm text-gray-500">
              Browse and inspect raw data files retrieved from Binance. Open,
              view, and analyze the contents directly within the panel to better
              understand the underlying market data.
            </p>
          </div>

          <div className="flex-1 overflow-y-auto h-full">
            <div className="grid grid-cols-6 gap-6 p-2">
              {files.map((file, index) => (
                <div
                  key={`${file.name}-${index}`}
                  onDoubleClick={() => handleDoubleClick(file)}
                  className="flex flex-col items-center cursor-pointer hover:opacity-80"
                >
                  <div className="p-4 rounded-lg bg-gray-100 dark:bg-gray-800 shadow-sm">
                    <FileText className="w-10 h-10 text-blue-500" />
                  </div>
                  <span className="mt-2 text-xs max-w-28 text-center break-words">
                    {file.name}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="min-w-[100vw] h-[100vh] flex flex-col">
          <DialogHeader>
            <DialogTitle>
                <span className="font-thin mr-2">
                    Showing first 100 rows of :
                </span>
                {selectedFile?.name}
            </DialogTitle>
          </DialogHeader>
          <div className="flex-1 overflow-auto px-2 py-2">
            {loadingPreview ? (
              <p className="text-gray-500">Loading preview...</p>
            ) : previewData ? (
              <div className="w-full overflow-auto">
                <table className="min-w-full text-sm border-collapse border border-gray-300 dark:border-gray-700">
                  <thead className="bg-gray-100 dark:bg-gray-800">
                    <tr>
                      {previewData.header?.map((col, idx) => (
                        <th
                          key={idx}
                          className="border border-gray-300 dark:border-gray-700 px-2 py-1 text-left"
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewData.preview.map((row, rowIdx) => (
                      <tr key={rowIdx} className="hover:bg-gray-50 dark:hover:bg-gray-900">
                        {row.map((cell, cellIdx) => (
                          <td
                            key={cellIdx}
                            className="border border-gray-300 dark:border-gray-700 px-2 py-1"
                          >
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>  
              </div>
            ) : (
              <p className="text-gray-500">No preview data available</p>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default FilePanel;
