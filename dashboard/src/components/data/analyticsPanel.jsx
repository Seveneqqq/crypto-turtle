import { useState } from "react"
import { Button } from "../components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogTrigger,
} from "../components/ui/dialog"

function AnalyticsPanel() {

  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(false)

  const loadReports = async () => {
    try {

      setLoading(true)

      const res = await fetch("http://localhost:8001/analytics/optuna/reports")
      const data = await res.json()

      setImages(data.reports || [])

    } catch (err) {
      console.error("Error loading reports", err)
    } finally {
      setLoading(false)
    }
  }

  const runAnalytics = async () => {
    try {

      await fetch("http://localhost:8001/analytics/optuna/dashboard", {
        method: "POST"
      })

      alert("Optuna dashboard generated")

    } catch (err) {
      console.error(err)
      alert("Failed generating dashboard")
    }
  }

  return (
    <div className="bg-card border rounded-xl p-4 space-y-4">

      <h2 className="text-xl font-bold">
        Optuna Analytics
      </h2>

      <Button
        className="w-full"
        onClick={runAnalytics}
      >
        Generate Dashboard
      </Button>

      <Dialog>

        <DialogTrigger asChild>

          <Button
            variant="secondary"
            className="w-full"
            onClick={loadReports}
          >
            Show Reports
          </Button>

        </DialogTrigger>

        <DialogContent className="!max-w-[95vw] !h-[95vh] overflow-y-scroll">

          <div className="text-xl font-bold mb-6">
            Optuna Reports
          </div>

          {loading && (
            <div>Loading reports...</div>
          )}

          <div className="grid grid-cols-3 gap-6">

            {images.map((img) => (

              <div
                key={img}
                className="border rounded-lg p-2 bg-muted"
              >

                <img
                  src={`http://localhost:8001/shared/files/optuna_reports/${img}`}
                  alt={img}
                  className="rounded-md"
                />

                <div className="text-xs mt-2 text-center">
                  {img}
                </div>

              </div>

            ))}

          </div>

        </DialogContent>

      </Dialog>

    </div>
  )
}

export default AnalyticsPanel