import React from 'react'
import FileSelect from '../../components/files/fileSelect'
import LeftSidebar from "../../components/leftSidebar.jsx"
import AccuracyPanel from '../../components/data/accuracyPanel.jsx'
import TasksHistoryPanel from '../../components/data/tasksHistoryPanel.jsx'

function AccuracyPage() {
  return (
     <LeftSidebar>
      <main className="w-full flex-1 px-6 py-2">
        <h1 className="text-3xl font-bold my-2">Welcome to the Accuracy Page!</h1>
          <div className="flex gap-6">
            <AccuracyPanel />
          </div>
            <TasksHistoryPanel action="calculate_accuracy" />
      </main>
    </LeftSidebar>
  )
}

export default AccuracyPage