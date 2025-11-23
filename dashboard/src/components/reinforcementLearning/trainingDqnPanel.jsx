import React, { useState } from "react";
import FileSelect from "../../components/files/fileSelect";
import { Card, CardHeader, CardContent, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";

export default function TrainingDqnPanel() {
  const [modelFile, setModelFile] = useState("");
  const [dataFile, setDataFile] = useState("");

  const [totalTimesteps, setTotalTimesteps] = useState(10000);
  const [learningRate, setLearningRate] = useState(0.0003);
  const [nSteps, setNSteps] = useState(2048);
  const [batchSize, setBatchSize] = useState(64);
  const [gamma, setGamma] = useState(0.99);
  const [entCoef, setEntCoef] = useState(0.0);

  const handleSubmit = async () => {
    if (!modelFile || !dataFile) return alert("Select model and data file first");

    const body = {
      model_file: modelFile,
      data_file: dataFile,
      total_timesteps: totalTimesteps,
      learning_rate: learningRate,
      n_steps: nSteps,
      batch_size: batchSize,
      gamma: gamma,
      ent_coef: entCoef,
    };

    try {
      const response = await fetch("http://localhost:8000/rl/train_agent_dqn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      console.log("Response:", data);
      alert("Training DQN started!");
    } catch (err) {
      console.error(err);
      alert("Error starting DQN training");
    }
  };

  return (
    <Card className="space-y-4">
      <CardHeader>
        <h2 className="text-lg font-semibold">Train DQN Agent</h2>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>Model File</Label>
          <FileSelect filesKey="models_ml" onChange={setModelFile} />
        </div>

        <div className="space-y-2">
          <Label>Data File</Label>
          <FileSelect filesKey="processed_rlagent_data" onChange={setDataFile} />
        </div>

        <div className="space-y-2">
          <Label>Total Timesteps</Label>
          <Input
            type="number"
            value={totalTimesteps}
            onChange={(e) => setTotalTimesteps(Number(e.target.value))}
          />
        </div>

        <div className="space-y-2">
          <Label>Learning Rate</Label>
          <Input
            type="number"
            step="0.0001"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
          />
        </div>

        <div className="space-y-2">
          <Label>n Steps</Label>
          <Input
            type="number"
            value={nSteps}
            onChange={(e) => setNSteps(Number(e.target.value))}
          />
        </div>

        <div className="space-y-2">
          <Label>Batch Size</Label>
          <Input
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
          />
        </div>

        <div className="space-y-2">
          <Label>Gamma</Label>
          <Input
            type="number"
            step="0.01"
            value={gamma}
            onChange={(e) => setGamma(parseFloat(e.target.value))}
          />
        </div>

        <div className="space-y-2">
          <Label>Ent Coef</Label>
          <Input
            type="number"
            step="0.01"
            value={entCoef}
            onChange={(e) => setEntCoef(parseFloat(e.target.value))}
          />
        </div>
      </CardContent>

      <CardFooter>
        <Button onClick={handleSubmit}>Start Training DQN</Button>
      </CardFooter>
    </Card>
  );
}
