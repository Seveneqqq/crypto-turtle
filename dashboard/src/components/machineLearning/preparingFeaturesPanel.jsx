import React, { useState } from "react";
import FileSelect from "../../components/files/fileSelect";
import { Card, CardHeader, CardContent, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";

export default function PreparingFeaturesPanel() {
  const [trainFile, setTrainFile] = useState("");
  const [testFile, setTestFile] = useState("");
  const [feVersion, setFeVersion] = useState("fe_v2");

  // Lista kolumn ogólnych
  const defaultColumns = ["open", "close", "high", "low", "volume"];
  const [columns, setColumns] = useState([...defaultColumns]);
  const [selectedColumns, setSelectedColumns] = useState([...defaultColumns]);
  const [newColumn, setNewColumn] = useState("");

  const toggleColumn = (col) => {
    setSelectedColumns(prev =>
      prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
    );
  };

  const addColumn = () => {
    const trimmed = newColumn.trim();
    if (!trimmed || columns.includes(trimmed)) return;
    setColumns(prev => [...prev, trimmed]);
    setSelectedColumns(prev => [...prev, trimmed]);
    setNewColumn("");
  };

  const handleSubmit = async () => {
    const body = {
      train_file: trainFile,
      test_file: testFile,
      fe_version: feVersion,
      feature_cols: selectedColumns,
    };
    try {
      const response = await fetch("http://localhost:8000/ml/prepare_features", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      console.log("Response:", data);
      alert("Features prepared successfully!");
    } catch (err) {
      console.error(err);
      alert("Error preparing features");
    }
  };

  return (
    <Card className="space-y-4">
      <CardHeader>
        <h2 className="text-lg font-semibold">Preparing Features</h2>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label>Train File</Label>
          <FileSelect filesKey="processed_trainmodel_data" onChange={setTrainFile} />
        </div>

        <div className="space-y-2">
          <Label>Test File</Label>
          <FileSelect filesKey="processed_testmodel_data" onChange={setTestFile} />
        </div>

        <div className="space-y-2">
          <Label>Feature Columns</Label>
          <div className="space-y-1">
            {columns.map(col => (
              <div key={col} className="flex items-center gap-2">
                <Checkbox
                  checked={selectedColumns.includes(col)}
                  onCheckedChange={() => toggleColumn(col)}
                />
                <span>{col}</span>
              </div>
            ))}
          </div>
          <div className="flex gap-2 mt-2">
            <Input
              placeholder="Add new column"
              value={newColumn}
              onChange={(e) => setNewColumn(e.target.value)}
            />
            <Button onClick={addColumn}>Add</Button>
          </div>
        </div>

        <div className="space-y-2">
          <Label>FE Version</Label>
          <Input value={feVersion} onChange={(e) => setFeVersion(e.target.value)} />
        </div>
      </CardContent>

      <CardFooter>
        <Button onClick={handleSubmit}>Prepare Features</Button>
      </CardFooter>
    </Card>
  );
}
