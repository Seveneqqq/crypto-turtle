"use client"

import React from "react"
import LeftSidebar from "../../components/leftSidebar.jsx"
import TasksHistoryPanel from "../../components/data/tasksHistoryPanel.jsx"
import TrainingPpoPanel from "../../components/reinforcementLearning/trainingPpoPanel.jsx"

function TrainingPpoPage() {
  return (
    <LeftSidebar>
      <main className="w-full flex-1 px-6 py-2">
        <h1 className="text-3xl font-bold my-2">Welcome to the Training PPO Page!</h1>
          <TrainingPpoPanel />
          <TasksHistoryPanel action="simulate" />
      </main>
    </LeftSidebar>
  )
}

export default TrainingPpoPage