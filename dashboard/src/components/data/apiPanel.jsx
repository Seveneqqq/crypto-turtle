import React, { useState, useEffect } from "react";
import { Card, CardContent } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "../ui/select";
import { Calendar } from "../ui/calendar";
import { toast, Toaster } from "sonner"
import { Popover, PopoverTrigger, PopoverContent } from "../ui/popover";
import { format } from "date-fns";

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

const intervals = ["1s", "15s", "1m", "5m", "15m", "1h", "4h", "1d"];

function ApiPanel() {
  const today = new Date();
  const oneMonthAgo = new Date();
  oneMonthAgo.setMonth(today.getMonth() - 1);

  const [symbol, setSymbol] = useState("BTCUSDT");
  const [interval, setInterval] = useState("1h");
  const [startDate, setStartDate] = useState(oneMonthAgo);
  const [endDate, setEndDate] = useState(today);
  const [filename, setFilename] = useState("");

  useEffect(() => {
    const monthsDiff =
      (endDate.getFullYear() - startDate.getFullYear()) * 12 +
      (endDate.getMonth() - startDate.getMonth());
    const defaultName = `${symbol}_${interval}_${monthsDiff}m_${format(today, "yyyy-MM-dd")}.csv`;
    setFilename(defaultName);
  }, [symbol, interval, startDate, endDate]);

  const handleSubmit = async () => {
    const url = `http://localhost:8000/data/get-historical-data?symbol=${symbol}&interval=${interval}&start_dt=${format(
      startDate,
      "yyyy-MM-dd HH:mm:ss"
    )}&end_dt=${format(endDate, "yyyy-MM-dd HH:mm:ss")}&filename=${filename}`;

    try {
      const res = await fetch(url)
      if (res.ok) {
        toast.success("Request sent successfully!", { position: "top-right" })
      } 
      else {
        toast.error("Error sending request", { position: "top-right" })
      }
    } catch (err) {
      toast.error("Failed to send request", { position: "top-right" })
      console.error(err)
    }
  };

  return (
    <div className="w-full max-w-2xl gap-6">
      <Toaster position="top-right" />
      <Card className="shadow-lg rounded-2xl">
        <CardContent className="space-y-4 px-6">
          <div className="space-y-1">
            <h2 className="text-xl font-bold tracking-tight">Historical Data Downloader</h2>
            <p className="text-sm text-gray-500">
              Select your preferred cryptocurrency, timeframe and date range to generate and download historical candlestick data.
            </p>
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium">Symbol</label>
            <Select value={symbol} onValueChange={setSymbol}>
              <SelectTrigger>
                <SelectValue placeholder="Select symbol" />
              </SelectTrigger>
              <SelectContent>
                {symbols.map((s) => (
                  <SelectItem key={s} value={s}>
                    {s}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium">Interval</label>
            <Select value={interval} onValueChange={setInterval}>
              <SelectTrigger>
                <SelectValue placeholder="Select interval" />
              </SelectTrigger>
              <SelectContent>
                {intervals.map((i) => (
                  <SelectItem key={i} value={i}>
                    {i}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-sm font-medium">Start Date</label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="w-full justify-start">
                    {format(startDate, "yyyy-MM-dd")}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="p-0">
                  <Calendar mode="single" selected={startDate} onSelect={setStartDate} />
                </PopoverContent>
              </Popover>
            </div>

            <div className="space-y-1">
              <label className="text-sm font-medium">End Date</label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="w-full justify-start">
                    {format(endDate, "yyyy-MM-dd")}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="p-0">
                  <Calendar mode="single" selected={endDate} onSelect={setEndDate} />
                </PopoverContent>
              </Popover>
            </div>
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium">Filename</label>
            <Input value={filename} onChange={(e) => setFilename(e.target.value)} />
            <p className="text-xs text-gray-400">Default format: SYMBOL_INTERVAL_RANGE_DATE.csv</p>
          </div>

          <Button className="w-full" onClick={handleSubmit}>
            Download Data
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

export default ApiPanel;