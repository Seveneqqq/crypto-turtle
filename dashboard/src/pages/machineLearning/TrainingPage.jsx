"use client"

import React from "react"
import LeftSidebar from "../../components/leftSidebar.jsx"
import TasksHistoryPanel from "../../components/data/tasksHistoryPanel.jsx"
import TrainingPanel from "../../components/machineLearning/trainingPanel.jsx"

function TrainingPage() {
  return (
    <LeftSidebar>
      <main className="w-full flex-1 px-6 py-2">
        <h1 className="text-3xl font-bold my-2">Welcome to the Training Page!</h1>
          <TrainingPanel />
          <TasksHistoryPanel action="simulate" />
      </main>
    </LeftSidebar>
  )
}

export default TrainingPage