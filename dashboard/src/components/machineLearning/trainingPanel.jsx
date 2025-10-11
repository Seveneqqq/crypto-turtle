import React, { useState } from "react";
import FileSelect from "../../components/files/fileSelect";
import { Card, CardHeader, CardTitle, CardContent } from "../ui/card";
import { Label } from "../ui/label";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "../ui/select";
import { Button } from "../ui/button";
import { ScrollArea } from "../ui/scroll-area";
import { Checkbox } from "../ui/checkbox";
import { toast } from "sonner";

function TrainingPanel() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [clfVersion, setClfVersion] = useState("c_v1");
  const [columns, setColumns] = useState([]);
  const [selectedCols, setSelectedCols] = useState([]);
  const [loadingCols, setLoadingCols] = useState(false);

    const fetchColumns = async (filename) => {
    if (!filename) {
      console.warn("No filename provided for fetching columns");
      return;
    }
    try {
      console.log("Fetching columns for file:", filename);
    
      setLoadingCols(true);
      setColumns([]);
      setSelectedCols([]);
    
      const res = await fetch(
        `http://localhost:8000/file/columns/processed_trainmodel_data/${filename}`,
        { credentials: "include" }
      );
      if (!res.ok) throw new Error("Failed to fetch columns");
      const data = await res.json();
    
      const cols = data.columns || [];
      setColumns(cols);
      setSelectedCols(cols); // ✅ automatycznie zaznacz wszystkie
    } catch (err) {
      console.error(err);
      toast.error("Error fetching file columns");
    } finally {
      setLoadingCols(false);
    }
  };

  const handleFileSelect = (file) => {
    console.log("Selected file:", file);
    if (!file) return;
    setSelectedFile(file);
    fetchColumns(file);
  };

  const handleColToggle = (col) => {
    setSelectedCols((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const handleTrain = async () => {
    if (!selectedFile) {
      toast.error("Please select a training file first");
      return;
    }

    if (selectedCols.length === 0) {
      toast.error("Please select at least one feature column");
      return;
    }

    try {
      const body = {
        df_filename: selectedFile,
        clf_version: clfVersion,
        feature_cols: selectedCols,
      };

      const res = await fetch("http://localhost:8000/ml/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(body),
      });

      if (!res.ok) throw new Error("Training failed");
      const data = await res.json();

      toast.success(`Training started! Model: ${data.model_name || clfVersion}`);
    } catch (err) {
      console.error(err);
      toast.error("Failed to start training");
    }
  };

  return (
    <div className="p-4 w-full">
      <Card className="shadow-md rounded-2xl">
        <CardHeader>
          <CardTitle>Train Machine Learning Model</CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* FILE SELECT */}
          <div>
            <Label>Select training file</Label>
            <FileSelect filesKey="processed_trainmodel_data" onChange={handleFileSelect} />
          </div>

          {/* CLASSIFIER VERSION */}
          <div>
            <Label>Classifier version</Label>
            <Select value={clfVersion} onValueChange={setClfVersion}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select classifier version" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="c_v1">c_v1</SelectItem>
                <SelectItem value="c_v2">c_v2</SelectItem>
                <SelectItem value="c_v3">c_v3</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* FEATURE COLUMNS */}
          <div>
            <Label>Feature columns</Label>
            {loadingCols ? (
              <p className="text-sm text-gray-500 mt-2">Loading columns...</p>
            ) : columns.length > 0 ? (
              <ScrollArea className="border rounded-md p-2 h-64 mt-2">
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                  {columns.map((col) => (
                    <div
                      key={col}
                      className="flex items-center space-x-2 cursor-pointer"
                      onClick={() => handleColToggle(col)}
                    >
                      <Checkbox checked={selectedCols.includes(col)} />
                      <span className="text-sm">{col}</span>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            ) : (
              <p className="text-sm text-gray-500 mt-2">No columns to display</p>
            )}
          </div>

          {/* TRAIN BUTTON */}
          <div className="pt-4 flex justify-end">
            <Button onClick={handleTrain} disabled={loadingCols}>
              {loadingCols ? "Loading..." : "Train"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default TrainingPanel;
