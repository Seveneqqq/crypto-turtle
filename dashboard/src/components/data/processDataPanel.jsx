import React, { useState } from "react"
import FileSelect from "../../components/files/fileSelect"
import { Button } from "../../components/ui/button"
import { Card, CardContent } from "../../components/ui/card"
import { Label } from "../../components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select"
import { Trash, ChevronUp, ChevronDown } from "lucide-react"

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

  const availableSteps = ["load_data", "make_features", "add_technical_indicators"]
  const [steps, setSteps] = useState(availableSteps)

  const availableMethods = ["RAW", "Z-SCORE", "ROBUST", "MINMAX", "ROLLING_Z"]
  const [preprocessingMethod, setPreprocessingMethod] = useState("RAW")

  const availableFolders = ["processed_trainmodel_data", "processed_testmodel_data", "processed_rlagent_data", "processed_simulation_data"]
  const [outputFolder, setOutputFolder] = useState("trainmodel")

  const availableSets = [
    "set1",
    "set2",
    "set3",
    "set4",

    "ma_5",
    "ma_10",
    "ma_20",
    "sma_10",
    "sma_30",
    "cumvol",
    "volume_norm",
    "std_20",

    "RSI_14",
    "EMA_20",
    "EMA_50",

    "BB_upper",
    "BB_lower",
    "Momentum_10",
    "ROC_10",

    "ATR_14",
    "VWAP",
    "ROC_5",
  ]

  const [technicalIndicatorsSets, setTechnicalIndicatorsSets] = useState([["set1"]])

  const removeStep = (index) => setSteps(steps.filter((_, i) => i !== index))

  const moveStep = (index, direction) => {
    const newSteps = [...steps]
    const swapIndex = direction === "up" ? index - 1 : index + 1
    if (swapIndex < 0 || swapIndex >= newSteps.length) return
    ;[newSteps[index], newSteps[swapIndex]] = [newSteps[swapIndex], newSteps[index]]
    setSteps(newSteps)
  }

  const toggleTechnicalSet = (groupIndex, setName) => {
    setTechnicalIndicatorsSets((prev) => {
      const newSets = [...prev]
      const group = newSets[groupIndex]

      if (group.includes(setName)) {
        newSets[groupIndex] = group.filter((s) => s !== setName)
      } else {
        newSets[groupIndex] = [...group, setName]
      }

      return newSets
    })
  }

  const addFeatureSet = () => {
    setTechnicalIndicatorsSets((prev) => [...prev, []])
  }

  const removeFeatureSet = (index) => {
    setTechnicalIndicatorsSets((prev) => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = () => {
    if (!filename) return alert("Select a file to process")
    if (!outputFolder) return alert("Select an output folder")

    const body = {
      filename,
      steps,
      preprocessing_method: preprocessingMethod,
      output_folders: [outputFolder],
      feature_sets: technicalIndicatorsSets,
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
          <FileSelect
            filesKey="binance_raw"
            value={filename}
            onChange={setFilename}
            placeholder="Select file"
          />
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
        </div>

        {/* TECHNICAL INDICATORS SETS */}
        <div>
          <Label>Technical Indicators Sets</Label>

          <div className="space-y-4 mt-2">

            {technicalIndicatorsSets.map((setGroup, groupIndex) => (

              <div key={groupIndex} className="border border-gray-700 rounded-md p-3 bg-gray-800">

                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium">
                    Feature Set {groupIndex + 1}
                  </span>

                  {technicalIndicatorsSets.length > 1 && (
                    <Button
                      size="icon"
                      variant="destructive"
                      onClick={() => removeFeatureSet(groupIndex)}
                    >
                      -
                    </Button>
                  )}
                </div>

                <div className="max-h-40 overflow-y-auto space-y-1">
                  {availableSets.map((set) => (
                    <label key={set} className="flex items-center gap-2 text-sm cursor-pointer">
                      <input
                        type="checkbox"
                        checked={setGroup.includes(set)}
                        onChange={() => toggleTechnicalSet(groupIndex, set)}
                        className="w-4 h-4 accent-blue-500"
                      />
                      {set}
                    </label>
                  ))}
                </div>

              </div>

            ))}

            <Button variant="outline" onClick={addFeatureSet}>
              + Add Feature Set
            </Button>

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

        {/* OUTPUT FOLDER */}
        <div>
          <Label>Output Folder</Label>
          <div className="flex flex-col gap-2 mt-2">
            {availableFolders.map((f) => (
              <div key={f} className="flex items-center gap-2">
                <input
                  type="radio"
                  id={f}
                  name="outputFolder"
                  value={f}
                  checked={outputFolder === f}
                  onChange={() => setOutputFolder(f)}
                  className="w-4 h-4 accent-blue-500"
                />
                <label htmlFor={f} className="text-sm cursor-pointer">
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