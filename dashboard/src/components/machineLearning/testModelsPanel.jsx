import React, { useState } from "react"
import FileSelect from "../../components/files/fileSelect"
import { Button } from "../ui/button"
import { Card, CardHeader, CardContent, CardTitle } from "../ui/card"
import { Loader2 } from "lucide-react"
import { toast, Toaster } from "sonner"

function TestModelsPanel() {
  const [modelFile, setModelFile] = useState(null)
  const [testFile, setTestFile] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleTestModel = async () => {
    if (!modelFile || !testFile) {
      toast.error("Please select both model and test file")
      return
    }

    setLoading(true)

    try {
      const res = await fetch("http://localhost:8000/ml/test_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_filename: modelFile,
          test_filename: testFile,
        }),
      })

      if (!res.ok) throw new Error(`Server responded with ${res.status}`)

      const data = await res.json()

      // ✅ Domyślny toast Sonner z treścią API
      toast.success(data.msg);
    } catch (error) {
      toast.error(
        `Request failed: ${error.message || "Unable to connect to API"}`
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {/* ✅ Domyślny Toaster Sonner */}
      <Toaster position="top-right"/>

      <Card className="p-6 space-y-6 shadow-lg rounded-2xl border">
        <CardHeader>
          <CardTitle className="text-lg font-semibold text-center">
            🧠 Test Machine Learning Model
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="space-y-2">
            <p className="font-medium">Select test dataset:</p>
            <FileSelect
              filesKey="processed_testmodel_data"
              onChange={setTestFile}
            />
          </div>

          <div className="space-y-2">
            <p className="font-medium">Select model file:</p>
            <FileSelect filesKey="models_ml" onChange={setModelFile} />
          </div>

          <Button
            onClick={handleTestModel}
            disabled={loading}
            className="w-full mt-4"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin mr-2 h-4 w-4" />
                Testing model...
              </>
            ) : (
              "Test model"
            )}
          </Button>
        </CardContent>
      </Card>
    </>
  )
}

export default TestModelsPanel
