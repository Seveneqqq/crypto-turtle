import React, { useState } from 'react';
import FileSelect from "../../components/files/fileSelect";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../ui/tabs";

function AccuracyPanel() {
  const [data, setData] = useState({ dir: '', file: '' });

  const handleSubmit = async () => {
    if (!data.file) {
      alert('Please select a data file!');
      return;
    }

    const payload = {
      data_dir: data.dir,
      data_file: data.file
    };

    try {
      const response = await fetch('http://localhost:8000/tests/calculate_quality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const result = await response.json();
      console.log(result);
      alert(result.msg);
    } catch (err) {
      console.error(err);
      alert('Error sending request');
    }
  };

  return (
    <Card className="w-full max-w-3xl my-4">
      <CardHeader>
        <CardTitle className="text-2xl font-bold">Calculate Accuracy</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">

        <div>
          <h3 className="text-xl font-semibold mb-2">Processed Data</h3>
          <Tabs defaultValue="processed">
            <TabsList className="mb-4">
              <TabsTrigger value="processed">Processed</TabsTrigger>
              <TabsTrigger value="rlagent">RL Agent</TabsTrigger>
              <TabsTrigger value="simulation">Simulation</TabsTrigger>
              <TabsTrigger value="testmodel">Test Model</TabsTrigger>
              <TabsTrigger value="trainmodel">Train Model</TabsTrigger>
            </TabsList>

            <TabsContent value="processed" className="space-y-4">
              <FileSelect
                placeholder="raw_data"
                filesKey="raw_data"
                value={data.file}
                onChange={(file) => setData({ dir: 'raw_data', file })}
              />
            </TabsContent>

            <TabsContent value="rlagent" className="space-y-4">
              <FileSelect
                placeholder="processed_rlagent_data"
                filesKey="processed_rlagent_data"
                value={data.file}
                onChange={(file) => setData({ dir: 'processed_rlagent_data', file })}
              />
            </TabsContent>

            <TabsContent value="simulation" className="space-y-4">
              <FileSelect
                placeholder="processed_simulation_data"
                filesKey="processed_simulation_data"
                value={data.file}
                onChange={(file) => setData({ dir: 'processed_simulation_data', file })}
              />
            </TabsContent>

            <TabsContent value="testmodel" className="space-y-4">
              <FileSelect
                placeholder="processed_testmodel_data"
                filesKey="processed_testmodel_data"
                value={data.file}
                onChange={(file) => setData({ dir: 'processed_testmodel_data', file })}
              />
            </TabsContent>

            <TabsContent value="trainmodel" className="space-y-4">
              <FileSelect
                placeholder="processed_trainmodel_data"
                filesKey="processed_trainmodel_data"
                value={data.file}
                onChange={(file) => setData({ dir: 'processed_trainmodel_data', file })}
              />
            </TabsContent>
          </Tabs>
        </div>

        <button
          onClick={handleSubmit}
          className="px-4 py-2 bg-blue-500 text-white rounded"
        >
          Send to API
        </button>
      </CardContent>
    </Card>
  );
}

export default AccuracyPanel;
