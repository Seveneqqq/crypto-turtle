import React from 'react'
import FileSelect from '../../components/files/fileSelect'
import LeftSidebar from "../../components/leftSidebar.jsx"
import ProcessDataPanel from '../../components/data/processDataPanel.jsx'
import TasksHistoryPanel from '../../components/data/tasksHistoryPanel.jsx'

function ProcessingDataPage() {
  return (
     <LeftSidebar>
      <main className="w-full flex-1 px-6 py-2">
        <h1 className="text-3xl font-bold my-2">Welcome to the Historical Data Page!</h1>
          <div className="flex gap-6">
            <ProcessDataPanel />
          </div>
            <TasksHistoryPanel action="process_data" />
      </main>
    </LeftSidebar>
  )
}

export default ProcessingDataPage