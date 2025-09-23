"use client"

import React from "react"
import LeftSidebar from "../../components/leftSidebar.jsx"
import SimulatePanel from "../../components/simulation/simulatePanel.jsx"
import TasksHistoryPanel from "../../components/data/tasksHistoryPanel.jsx"

export default function SimulationPage() {
  return (
    <LeftSidebar>
      <main className="w-full flex-1 px-6 py-2">
        <h1 className="text-3xl font-bold my-2">Welcome to the Simulation Page!</h1>
          <SimulatePanel />
          <TasksHistoryPanel action="simulate" />
      </main>
    </LeftSidebar>
  )
}
