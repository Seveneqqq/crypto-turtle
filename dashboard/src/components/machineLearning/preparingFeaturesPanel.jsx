import React from 'react'

import FileSelect from "../../components/files/fileSelect"

function PreparingFeaturesPanel() {
  return (
    <>
      <FileSelect filesKey="processed_testmodel_data" />
      <FileSelect filesKey="processed_trainmodel_data" />
    </>
  )
}

export default PreparingFeaturesPanel