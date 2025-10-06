import React, { useState } from "react"
import FileSelect from "../../components/files/fileSelect"
import { Button } from "../../components/ui/button"
import { Card, CardContent } from "../../components/ui/card"
import { Label } from "../../components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select"
import { Trash, ChevronUp, ChevronDown } from "lucide-react"
import { Checkbox } from "../../components/ui/checkbox"

function ListItem({ item, onRemove, onMoveUp, onMoveDown, isFirst, isLast }) {
  return (
    <div className="flex items-center justify-between bg-gray-800 text-gray-100 rounded-lg p-2 mb-2 shadow-sm">
      <span>{item}</span>
      <div className="flex items-center gap-2">
        <Button size="icon" variant="outline" onClick={onMoveUp} disabled={isFirst}>
          <ChevronUp className="h-4 w-4" />
        </Button>
        <Button size="icon" variant="outline" onClick={onMoveDown} disabled={isLast}>
          <ChevronDown className="h-4 w-4" />
        </Button>
        <Button size="icon" variant="destructive" onClick={onRemove}>
          <Trash className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}

export default function ProcessDataPanel() {
  const [filename, setFilename] = useState("")

  const availableSteps = ["load_data", "make_features", "add_technical_indicators", "feature_engineering", "custom_step"]
  const [steps, setSteps] = useState(availableSteps.slice(0, 3))

  const availableMethods = ["RAW", "Z-SCORE", "ROBUST", "MINMAX", "ROLLING_Z"]
  const [preprocessingMethod, setPreprocessingMethod] = useState("RAW")

  const availableFolders = ["trainmodel", "testmodel", "rlagent", "simulation"]
  const [outputFolders, setOutputFolders] = useState(["trainmodel"])

  // KROKI
  const addStep = (step) => {
    if (!steps.includes(step)) setSteps([...steps, step])
  }
  const removeStep = (index) => setSteps(steps.filter((_, i) => i !== index))
  const moveStep = (index, direction) => {
    const newSteps = [...steps]
    const swapIndex = direction === "up" ? index - 1 : index + 1
    if (swapIndex < 0 || swapIndex >= newSteps.length) return
    ;[newSteps[index], newSteps[swapIndex]] = [newSteps[swapIndex], newSteps[index]]
    setSteps(newSteps)
  }

  // FOLDERY
  const toggleFolder = (folder) => {
    setOutputFolders((prev) =>
      prev.includes(folder) ? prev.filter((f) => f !== folder) : [...prev, folder]
    )
  }

  const handleSubmit = () => {
    if (!filename) return alert("Select a file to process")
    if (outputFolders.length === 0) return alert("Select at least one output folder")

    const body = {
      filename,
      steps,
      preprocessing_method: preprocessingMethod, // ✅ backend spójny
      output_folders: outputFolders,
    }

    fetch("http://localhost:8000/data/process-data", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.detail) {
          alert(`❌ Error: ${JSON.stringify(data.detail)}`)
        } else {
          alert(`✅ ${data.msg}`)
        }
      })
      .catch((err) => console.error("Error:", err))
  }

  return (
    <Card className="px-6 py-4 max-w-3xl space-y-6 bg-gray-900 text-gray-100">
      <CardContent className="space-y-6">
        {/* FILE */}
        <div>
          <Label className="mb-3">Select File</Label>
          <FileSelect filesKey="binance_raw" value={filename} onChange={setFilename} placeholder="Select file" />
        </div>

        {/* STEPS */}
        <div>
          <Label>Processing Steps</Label>
          <div className="mt-2">
            {steps.map((s, i) => (
              <ListItem
                key={s}
                item={s}
                onRemove={() => removeStep(i)}
                onMoveUp={() => moveStep(i, "up")}
                onMoveDown={() => moveStep(i, "down")}
                isFirst={i === 0}
                isLast={i === steps.length - 1}
              />
            ))}
          </div>
          <div className="flex gap-2 mt-3">
            <Select onValueChange={addStep}>
              <SelectTrigger className="w-64">
                <SelectValue placeholder="Add step..." />
              </SelectTrigger>
              <SelectContent>
                {availableSteps
                  .filter((s) => !steps.includes(s)) // ✅ unikamy duplikatów
                  .map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* PREPROCESSING */}
        <div>
          <Label>Preprocessing Method</Label>
          <div className="mt-2 w-64">
            <Select value={preprocessingMethod} onValueChange={setPreprocessingMethod}>
              <SelectTrigger className="w-64">
                <SelectValue placeholder="Select method..." />
              </SelectTrigger>
              <SelectContent>
                {availableMethods.map((m) => (
                  <SelectItem key={m} value={m}>
                    {m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* FOLDERY */}
        <div>
          <Label>Output Folders</Label>
          <div className="flex flex-col gap-2 mt-2">
            {availableFolders.map((f) => (
              <div key={f} className="flex items-center gap-2">
                <Checkbox id={f} checked={outputFolders.includes(f)} onCheckedChange={() => toggleFolder(f)} />
                <label htmlFor={f} className="text-sm">
                  {f}
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* SUBMIT */}
        <Button className="w-full mt-4" onClick={handleSubmit}>
          🚀 Process Data
        </Button>
      </CardContent>
    </Card>
  )
}
