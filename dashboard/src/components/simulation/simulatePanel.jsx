import React, { useEffect, useState } from "react";
import { Card, CardContent } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Slider } from "../ui/slider";
import { Plus, Trash } from "lucide-react";

const symbols = [
  "BTCUSDT",
  "ETHUSDT",
  "BNBUSDT",
  "XRPUSDT",
  "ADAUSDT",
  "SOLUSDT",
  "DOGEUSDT",
  "TRXUSDT",
  "DOTUSDT",
  "MATICUSDT",
];

export default function SimulatePanel() {
  const [balance, setBalance] = useState(1000);
  const [timeValue, setTimeValue] = useState(1);
  const [timeUnit, setTimeUnit] = useState("s");
  const [thresholdBuy, setThresholdBuy] = useState([0.6]);
  const [thresholdSell, setThresholdSell] = useState([0.6]);
  const [files, setFiles] = useState([]);
  const [models, setModels] = useState([]);
  const [pairs, setPairs] = useState([
    { symbol: "BTCUSDT", file: "", model: "" }
  ]);

  useEffect(() => {
    fetch("http://localhost:8000/file/binance", { credentials: "include" })
      .then(res => res.json())
      .then(data => setFiles(data));

    fetch("http://localhost:8000/file/models", { credentials: "include" })
      .then(res => res.json())
      .then(data => setModels(data));
  }, []);

  const handleAddPair = () => {
    setPairs([...pairs, { symbol: symbols[0], file: "", model: "" }]);
  };

  const handleRemovePair = (index) => {
    setPairs(pairs.filter((_, i) => i !== index));
  };

  const handleChangePair = (index, key, value) => {
    const newPairs = [...pairs];
    newPairs[index][key] = value;
    setPairs(newPairs);
  };

  const handleSubmit = () => {
    const data_files = {};
    const models_paths = {};
    pairs.forEach(p => {
      if (p.file && p.model) {
        data_files[p.symbol.replace("USDT", "")] = p.file;
        models_paths[p.symbol.replace("USDT", "")] = p.model;
      }
    });

    const time_step =
      timeUnit === "s"
        ? timeValue
        : timeUnit === "m"
        ? timeValue * 60
        : timeValue * 3600;

    const body = {
      balance_start: balance,
      time_step,
      data_files,
      models: models_paths,
      threshold_buy: thresholdBuy[0],
      threshold_sell: thresholdSell[0],
    };

    fetch("http://localhost:8000/tests/simulate", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(res => res.json()).then(console.log);
  };

  return (
    <Card className="p-4 max-w-2xl space-y-6">
      <CardContent className="space-y-6">
        <div>
          <Label>Balance Start</Label>
          <Input
            type="number"
            min={100}
            max={25000}
            value={balance}
            onChange={e => setBalance(Number(e.target.value))}
          />
        </div>

        <div className="flex gap-2 items-end">
          <div className="flex-1">
            <Label>Time Step</Label>
            <Input
              type="number"
              min={1}
              value={timeValue}
              onChange={e => setTimeValue(Number(e.target.value))}
            />
          </div>
          <div>
            <Select value={timeUnit} onValueChange={setTimeUnit}>
              <SelectTrigger className="w-[120px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="s">Seconds</SelectItem>
                <SelectItem value="m">Minutes</SelectItem>
                <SelectItem value="h">Hours</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div>
          <Label>Threshold Buy ({thresholdBuy[0].toFixed(2)})</Label>
          <Slider
            value={thresholdBuy}
            min={0}
            max={1}
            step={0.01}
            onValueChange={setThresholdBuy}
          />
        </div>

        <div>
          <Label>Threshold Sell ({thresholdSell[0].toFixed(2)})</Label>
          <Slider
            value={thresholdSell}
            min={0}
            max={1}
            step={0.01}
            onValueChange={setThresholdSell}
          />
        </div>

        <div className="space-y-4">
          <Label>Pairs</Label>
          {pairs.map((pair, index) => (
            <div key={index} className="flex gap-2 items-center">
              <Select
                value={pair.symbol}
                onValueChange={val => handleChangePair(index, "symbol", val)}
              >
                <SelectTrigger className="w-[120px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {symbols.map(s => (
                    <SelectItem key={s} value={s}>{s}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={pair.file}
                onValueChange={val => handleChangePair(index, "file", val)}
              >
                <SelectTrigger className="w-[250px]">
                  <SelectValue placeholder="Select file" />
                </SelectTrigger>
                <SelectContent>
                  {files.map(f => (
                    <SelectItem key={f.name} value={f.name}>{f.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={pair.model}
                onValueChange={val => handleChangePair(index, "model", val)}
              >
                <SelectTrigger className="w-[250px]">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {models.map(m => (
                    <SelectItem key={m.name} value={m.name}>{m.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Button variant="destructive" size="icon" onClick={() => handleRemovePair(index)}>
                <Trash className="h-4 w-4" />
              </Button>
            </div>
          ))}
          <Button onClick={handleAddPair} variant="secondary">
            <Plus className="h-4 w-4 mr-2" /> Add Pair
          </Button>
        </div>

        <Button className="w-full" onClick={handleSubmit}>
          Run Simulation
        </Button>
      </CardContent>
    </Card>
  );
}
