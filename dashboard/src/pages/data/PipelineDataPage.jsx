import React, { useState } from 'react'
import FileSelect from '../../components/files/fileSelect'
import LeftSidebar from "../../components/leftSidebar.jsx"
import ProcessDataPanel from '../../components/data/processDataPanel.jsx'
import TasksHistoryPanel from '../../components/data/tasksHistoryPanel.jsx'
import DataOperationsPanel from '../../components/data/dataOperationsPanel.jsx'

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogTrigger
} from "@/components/ui/dialog"

function PipelineDataPage() {

  const [reports, setReports] = useState([])
  const [loadingReports, setLoadingReports] = useState(false)

  const [selectedImage, setSelectedImage] = useState(null)
  const [openImageViewer, setOpenImageViewer] = useState(false)

  const fetchReports = async () => {
    try {
      setLoadingReports(true)

      const res = await fetch("http://localhost:8001/analytics/optuna/reports")
      const data = await res.json()

      setReports(data.reports || [])
    } catch (err) {
      console.error("Failed loading reports", err)
    } finally {
      setLoadingReports(false)
    }
  }

  const generateDashboard = async () => {
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

  const openImage = (file) => {
    setSelectedImage(file)
    setOpenImageViewer(true)
  }

  return (
     <LeftSidebar>
      <main className="w-full flex-1 px-8 py-2">

        <h1 className="text-3xl font-bold my-2">
          Welcome to the Pipeline Data Page!
        </h1>

        <div className="flex gap-3">

          <div className="flex-1">
            <DataOperationsPanel />
          </div>

          <div className="w-auto h-115 bg-card border rounded-xl p-10 mt-46 space-y-4">

            <h2 className="text-xl font-bold">
              Optuna Analytics
            </h2>

            <Button
              className="w-full"
              onClick={generateDashboard}
            >
              Generate Dashboard
            </Button>

            <Dialog>

              <DialogTrigger asChild>

                <Button
                  variant="secondary"
                  className="w-full"
                  onClick={fetchReports}
                >
                  Show charts
                </Button>

              </DialogTrigger>

              <DialogContent className="!max-w-full !w-full !h-full overflow-y-scroll">

                <div className="text-xl font-bold mb-6">
                  Optuna Reports
                </div>

                {loadingReports && (
                  <div>Loading reports...</div>
                )}

                <div className="grid grid-cols-3 gap-6">

                  {reports.map((file) => (

                    <div
                      key={file}
                      className="border rounded-lg p-2 bg-muted cursor-pointer hover:scale-[1.02] transition"
                      onClick={() => openImage(file)}
                    >

                      <img
                        src={`http://localhost:8001/shared/files/optuna_reports/${file}`}
                        alt={file}
                        className="rounded-md"
                      />

                      <div className="text-xs mt-2 text-center">
                        {file}
                      </div>

                    </div>

                  ))}

                </div>

              </DialogContent>

            </Dialog>

          </div>

        </div>

        <TasksHistoryPanel action="process_data"/>

        {/* MODAL POWIĘKSZONEGO OBRAZU */}

        <Dialog open={openImageViewer} onOpenChange={setOpenImageViewer}>

          <DialogContent className="!max-w-full !w-full !h-full flex items-center justify-center bg-black/90">

            {selectedImage && (

              <div className="relative w-full h-full flex items-center justify-center">

                <button
                  onClick={() => setOpenImageViewer(false)}
                  className="absolute top-6 right-6 text-white text-3xl font-bold"
                >
                  ✕
                </button>

                <img
                  src={`http://localhost:8001/shared/files/optuna_reports/${selectedImage}`}
                  alt={selectedImage}
                  className="max-h-[90vh] max-w-[95vw] object-contain"
                />

              </div>

            )}

          </DialogContent>

        </Dialog>

      </main>
    </LeftSidebar>
  )
}

export default PipelineDataPage