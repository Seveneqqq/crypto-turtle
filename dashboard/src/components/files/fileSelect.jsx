import React, { useEffect, useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";

export default function FileSelect({
  filesKey,
  value,
  onChange,
  placeholder = "Select file",
  className = "w-[250px]"
}) {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!filesKey) {
      setFiles([]);
      return;
    }

    setLoading(true);
    setError(null);

    fetch(`http://localhost:8000/file/files-by-key/${encodeURIComponent(filesKey)}`, {
      credentials: "include"
    })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (data && Array.isArray(data.files)) {
          setFiles(data.files);
        } else {
          // Jeśli backend zwraca inny format — zabezpieczenie
          setFiles([]);
        }
      })
      .catch(err => {
        console.error("Failed to fetch files for key", filesKey, err);
        setError("Cannot load files");
        setFiles([]);
      })
      .finally(() => setLoading(false));
  }, [filesKey]);

  return (
    <div>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className={className}>
          <SelectValue placeholder={loading ? "Loading..." : error ? error : placeholder} />
        </SelectTrigger>
        <SelectContent>
          {files.length === 0 && !loading ? (
            <SelectItem value="__empty__" disabled>
              {error ? error : "No files"}
            </SelectItem>
          ) : (
            files.map(f => (
              <SelectItem key={f.name} value={f.name}>
                {f.name}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
    </div>
  );
}
