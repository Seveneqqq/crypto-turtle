import React from 'react'

import FileSelect from "../../components/files/fileSelect"

function PreparingFeaturesPanel() {
  return (
    <>
      <FileSelect key="processed_testmodel_data" />
      <FileSelect key="processed_trainmodel_data" />
    </>
  )
}

export default PreparingFeaturesPanel