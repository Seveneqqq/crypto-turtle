"use client"

import React from "react"
import LeftSidebar from "../components/leftSidebar.jsx"
import SimulatePanel from "../components/simulation/simulatePanel.jsx"
import TasksHistoryPanel from "../components/data/tasksHistoryPanel.jsx"
import ApiPanel from "../components/data/apiPanel.jsx"
import FilePanel from "../components/data/filePanel.jsx"

export default function HomePage() {
  return (
    <LeftSidebar>
      <main className="w-full flex-1 px-6 py-2">
        <h1 className="text-3xl font-bold my-2">Welcome to the Home Page!</h1>
          <div className="flex gap-6">
            <ApiPanel />
            <FilePanel />
          </div>
          <TasksHistoryPanel action="get_historical_data" />
      </main>
    </LeftSidebar>
  )
}
