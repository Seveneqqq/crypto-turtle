"use client"

import React from "react"
import LeftSidebar from "../../components/leftSidebar.jsx"
import TasksHistoryPanel from "../../components/data/tasksHistoryPanel.jsx"
import PreparingFeaturesPanel from "../../components/machineLearning/preparingFeaturesPanel.jsx"

function PreparingFeaturesPage() {
  return (
    <LeftSidebar>
      <main className="w-full flex-1 px-6 py-2">
        <h1 className="text-3xl font-bold my-2">Welcome to the Preparing Features Page!</h1>
          <PreparingFeaturesPanel />
          <TasksHistoryPanel action="simulate" />
      </main>
    </LeftSidebar>
  )
}

export default PreparingFeaturesPage