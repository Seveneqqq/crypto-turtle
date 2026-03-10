import { useState, useEffect } from "react"

import FileSelect from "../../components/files/fileSelect"
import { Trash, ChevronUp, ChevronDown } from "lucide-react"

import { Button } from "../../components/ui/button"
import { Input } from "../../components/ui/input"
import { Label } from "../../components/ui/label"
import { Calendar } from "../../components/ui/calendar"
import { Popover, PopoverTrigger, PopoverContent } from "../../components/ui/popover"

import { toast, Toaster } from "sonner"
import { format } from "date-fns"

import {
Tabs,
TabsList,
TabsTrigger,
TabsContent
} from "../../components/ui/tabs"

import {
Card,
CardHeader,
CardTitle,
CardContent
} from "../../components/ui/card"

import {
Select,
SelectTrigger,
SelectValue,
SelectContent,
SelectItem
} from "../../components/ui/select"

const API="http://localhost:8000/data"



/* ================= OPTIONS ================= */

const symbols=[
"BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT",
"ADAUSDT","SOLUSDT","DOGEUSDT","TRXUSDT",
"DOTUSDT","MATICUSDT"
]

const intervals=[
"1s","15s","1m","5m","15m","1h","4h","1d"
]

const stepOptions=[
"load_data",
"make_features",
"add_technical_indicators"
]

const preprocessingMethods=[
"RAW","Z-SCORE","ROBUST","MINMAX","ROLLING_Z"
]

const outputFolders=[
"processed_trainmodel_data",
"processed_testmodel_data",
"processed_rlagent_data",
"processed_simulation_data"
]

const outputFoldersLstm=[
"processed_trainmodel_data_lstm",
"processed_testmodel_data_lstm",
"processed_rlagent_data_lstm",
"processed_simulation_data_lstm"
]

const horizonOptions=[1,3,5,10,20,30,40]

const featureOptions=[

"set1","set2","set3","set4",

"ma_5","ma_10","ma_20",
"sma_10","sma_30",
"cumvol","volume_norm","std_20",

"RSI_14","EMA_20","EMA_50",

"BB_upper","BB_lower",
"Momentum_10","ROC_10",

"ATR_14","VWAP","ROC_5"

]



/* ================= STEP LIST ITEM ================= */

function ListItem({item,onRemove,onMoveUp,onMoveDown,isFirst,isLast}){

return(

<div className="flex items-center justify-between bg-gray-800 text-gray-100 rounded-lg p-2 mb-2 shadow-sm">

<span>{item}</span>

<div className="flex items-center gap-2">

<Button size="icon" variant="outline" onClick={onMoveUp} disabled={isFirst}>
<ChevronUp className="h-4 w-4"/>
</Button>

<Button size="icon" variant="outline" onClick={onMoveDown} disabled={isLast}>
<ChevronDown className="h-4 w-4"/>
</Button>

<Button size="icon" variant="destructive" onClick={onRemove}>
<Trash className="h-4 w-4"/>
</Button>

</div>

</div>

)

}



export default function DataOperationsPanel(){

const [loading,setLoading]=useState(false)



/* ================= DOWNLOAD ================= */

const today=new Date()
const oneMonthAgo=new Date()
oneMonthAgo.setMonth(today.getMonth()-1)

const [symbol,setSymbol]=useState("BTCUSDT")
const [interval,setInterval]=useState("1h")

const [startDate,setStartDate]=useState(oneMonthAgo)
const [endDate,setEndDate]=useState(today)

const [filename,setFilename]=useState("")

useEffect(()=>{

const monthsDiff=
(endDate.getFullYear()-startDate.getFullYear())*12+
(endDate.getMonth()-startDate.getMonth())

const defaultName=
`${symbol}_${interval}_${monthsDiff}m_${format(today,"yyyy-MM-dd")}.csv`

setFilename(defaultName)

},[symbol,interval,startDate,endDate])


const download=async()=>{

const url=
`${API}/get-historical-data?symbol=${symbol}&interval=${interval}&start_dt=${format(startDate,"yyyy-MM-dd HH:mm:ss")}&end_dt=${format(endDate,"yyyy-MM-dd HH:mm:ss")}&filename=${filename}`

try{

const res=await fetch(url)

if(res.ok){

toast.success("Request sent successfully!")

}else{

toast.error("Error sending request")

}

}catch(err){

toast.error("Failed to send request")
console.error(err)

}

}



/* ================= PROCESS DATA ================= */

const [dataset,setDataset]=useState("")
const [steps,setSteps]=useState(stepOptions)
const [preprocessing,setPreprocessing]=useState("RAW")
const [outputFolder,setOutputFolder]=useState("processed_trainmodel_data")
const [horizon,setHorizon]=useState(5)

const [technicalIndicatorsSets,setTechnicalIndicatorsSets]=useState([["set1"]])



const removeStep=(index)=>{
setSteps(steps.filter((_,i)=>i!==index))
}

const moveStep=(index,direction)=>{

const newSteps=[...steps]

const swapIndex=direction==="up"
? index-1
: index+1

if(swapIndex<0 || swapIndex>=newSteps.length) return

;[newSteps[index],newSteps[swapIndex]]=
[newSteps[swapIndex],newSteps[index]]

setSteps(newSteps)

}



const toggleTechnicalSet=(groupIndex,setName)=>{

setTechnicalIndicatorsSets(prev=>{

const newSets=[...prev]
const group=newSets[groupIndex]

if(group.includes(setName)){

newSets[groupIndex]=group.filter(s=>s!==setName)

}else{

newSets[groupIndex]=[...group,setName]

}

return newSets

})

}



const addFeatureSet=()=>{
setTechnicalIndicatorsSets(prev=>[...prev,[]])
}

const removeFeatureSet=(index)=>{
setTechnicalIndicatorsSets(prev=>prev.filter((_,i)=>i!==index))
}



const processData=()=>{

if(!dataset) return alert("Select file")

const body={

filename:dataset,
steps,
preprocessing_method:preprocessing,
output_folders:[outputFolder],
feature_sets:technicalIndicatorsSets,
horizon

}

fetch(`${API}/process-data`,{

method:"POST",
headers:{ "Content-Type":"application/json" },
body:JSON.stringify(body)

})
.then(res=>res.json())
.then(data=>{

if(data.detail){

alert(JSON.stringify(data.detail))

}else{

toast.success("Processing request sent")

}

})

}



/* ================= PROCESS LSTM ================= */

const [lstmDataset,setLstmDataset]=useState("")
const [lstmSteps,setLstmSteps]=useState(["load_data"])
const [lstmPreprocessing,setLstmPreprocessing]=useState("RAW")
const [lstmOutputFolder,setLstmOutputFolder]=useState("processed_trainmodel_data_lstm")

const [lstmFeatureSets,setLstmFeatureSets]=useState([["set1"]])

const toggleLSTMTechnicalSet=(groupIndex,setName)=>{

setLstmFeatureSets(prev=>{

const newSets=[...prev]
const group=newSets[groupIndex]

if(group.includes(setName)){

newSets[groupIndex]=group.filter(s=>s!==setName)

}else{

newSets[groupIndex]=[...group,setName]

}

return newSets

})

}

const addLSTMFeatureSet=()=>{
setLstmFeatureSets(prev=>[...prev,[]])
}

const removeLSTMFeatureSet=(index)=>{
setLstmFeatureSets(prev=>prev.filter((_,i)=>i!==index))
}

const toggleLSTMFeature=(f)=>{

setLstmFeatures(prev=>
prev.includes(f)
? prev.filter(x=>x!==f)
: [...prev,f]
)

}

const processLSTM=()=>{

if(!lstmDataset) return alert("Select file")

const body={

filename:lstmDataset,

steps:lstmSteps,

preprocessing_method:lstmPreprocessing,

output_folders:[lstmOutputFolder],

technical_indicators_set:"set1",

selected_features:lstmFeatureSets.flat()

}

fetch(`${API}/process-data-lstm`,{

method:"POST",
headers:{ "Content-Type":"application/json" },

body:JSON.stringify(body)

})
.then(res=>res.json())
.then(data=>{

if(data.detail){

alert(JSON.stringify(data.detail))

}else{

toast.success("LSTM dataset processing started")

}

})

}



/* ================= TUNE ML ================= */
const [tuneFiles,setTuneFiles]=useState("")
const [trials,setTrials]=useState(20)
const [maxFeatures,setMaxFeatures]=useState(2)
const [topK,setTopK]=useState(5)
const [metric,setMetric]=useState("mse")

const tuneData=()=>{

fetch(`${API}/tune-data`,{

method:"POST",
headers:{ "Content-Type":"application/json" },

body:JSON.stringify({

action:"tune_data",

filename_candidates:tuneFiles.split(",").map(x=>x.trim()),

steps:stepOptions,

n_trials:trials,

technical_sets:["combinations"],

preprocessing_methods:preprocessingMethods,

output_folders:["processed_trainmodel_data"],

metrics:[metric],

max_features_in_combo:maxFeatures,

top_k_combinations:topK

})

})

toast.success("Optuna ML tuning started")

}


/* ================= TUNE LSTM ================= */
const [seqLen,setSeqLen]=useState(32)
const [lstmUnits,setLstmUnits]=useState(32)
const [lstmLayers,setLstmLayers]=useState(1)
const [scalerType,setScalerType]=useState("standard")
const [epochs,setEpochs]=useState(1)

const tuneLSTM=()=>{

fetch(`${API}/tune-data-lstm`,{

method:"POST",
headers:{ "Content-Type":"application/json" },

body:JSON.stringify({

action:"tune_lstm",

filename_candidates:tuneFiles.split(",").map(x=>x.trim()),

steps:["load_data"],

n_trials:trials,

technical_sets:["combinations"],

preprocessing_methods:["RAW"],

target_col:"close",

seq_len_range:[seqLen,seqLen,1],

scaler_types:[scalerType],

lstm_units_range:[lstmUnits,lstmUnits,1],

lstm_layers_range:[lstmLayers,lstmLayers],

dropout_range:[0.1,0.1,0.1],

lr_range:[0.001,0.001],

batch_sizes:[32],

epochs,

max_features_in_combo:5,

top_k_combinations:5

})

})

toast.success("Optuna LSTM tuning started")

}

/* ================= UI ================= */

return(

<div className="p-10 pl-1 space-y-8">

<Toaster position="top-right"/>

<h1 className="text-3xl font-bold">
ML Data Pipeline
</h1>



<Tabs defaultValue="download">

<TabsList className="grid grid-cols-5">

<TabsTrigger value="download">Download</TabsTrigger>
<TabsTrigger value="process">Process</TabsTrigger>
<TabsTrigger value="lstm">Process LSTM</TabsTrigger>
<TabsTrigger value="tune">Tune ML</TabsTrigger>
<TabsTrigger value="tune-lstm">Tune LSTM</TabsTrigger>

</TabsList>



{/* ================= DOWNLOAD ================= */}
{/* (bez zmian) */}


{/* ================= DOWNLOAD ================= */}

<TabsContent value="download">

<Card className="shadow-lg rounded-2xl">

<CardHeader>
<CardTitle>Historical Data Downloader</CardTitle>
</CardHeader>

<CardContent className="space-y-6 px-6">

<div>

<Label className="mb-2 block">Symbol</Label>

<Select value={symbol} onValueChange={setSymbol}>

<SelectTrigger>
<SelectValue/>
</SelectTrigger>

<SelectContent>

{symbols.map(s=>(
<SelectItem key={s} value={s}>{s}</SelectItem>
))}

</SelectContent>

</Select>

</div>


<div>

<Label className="mb-2 block">Interval</Label>

<Select value={interval} onValueChange={setInterval}>

<SelectTrigger>
<SelectValue/>
</SelectTrigger>

<SelectContent>

{intervals.map(i=>(
<SelectItem key={i} value={i}>{i}</SelectItem>
))}

</SelectContent>

</Select>

</div>



<div className="grid grid-cols-2 gap-4">

<div>

<Label className="mb-2 block">Start Date</Label>

<Popover>

<PopoverTrigger asChild>

<Button variant="outline" className="w-full justify-start">
{format(startDate,"yyyy-MM-dd")}
</Button>

</PopoverTrigger>

<PopoverContent className="p-0">

<Calendar
mode="single"
selected={startDate}
onSelect={setStartDate}
/>

</PopoverContent>

</Popover>

</div>


<div>

<Label className="mb-2 block">End Date</Label>

<Popover>

<PopoverTrigger asChild>

<Button variant="outline" className="w-full justify-start">
{format(endDate,"yyyy-MM-dd")}
</Button>

</PopoverTrigger>

<PopoverContent className="p-0">

<Calendar
mode="single"
selected={endDate}
onSelect={setEndDate}
/>

</PopoverContent>

</Popover>

</div>

</div>



<div>

<Label className="mb-2 block">Filename</Label>

<Input
value={filename}
onChange={e=>setFilename(e.target.value)}
/>

</div>



<Button className="w-full" onClick={download}>
Download Data
</Button>

</CardContent>

</Card>

</TabsContent>



{/* ================= PROCESS ================= */}

<TabsContent value="process">

<Card>

<CardHeader>
<CardTitle>Process Dataset</CardTitle>
</CardHeader>

<CardContent className="space-y-6">

<div>

<Label className="mb-3 block">Dataset</Label>

<FileSelect
filesKey="raw_data"
value={dataset}
onChange={setDataset}
/>

</div>

<div>

<Label className="mb-3 block">
Prediction Horizon
</Label>

<Select value={String(horizon)} onValueChange={(v)=>setHorizon(Number(v))}>

<SelectTrigger className="w-40">
<SelectValue placeholder="Select horizon"/>
</SelectTrigger>

<SelectContent>

{horizonOptions.map(h=>(
<SelectItem key={h} value={String(h)}>
{h}
</SelectItem>
))}

</SelectContent>

</Select>

<p className="text-xs text-gray-400 mt-2">
Number of future candles used to generate the target label
</p>

</div>

<div>

<Label className="mb-3 block">Processing Steps</Label>

{steps.map((s,i)=>(
<ListItem
key={s}
item={s}
onRemove={()=>removeStep(i)}
onMoveUp={()=>moveStep(i,"up")}
onMoveDown={()=>moveStep(i,"down")}
isFirst={i===0}
isLast={i===steps.length-1}
/>
))}

</div>


<div>

<Label className="mb-3 block">Feature Sets</Label>

{technicalIndicatorsSets.map((setGroup,groupIndex)=>(

<div key={groupIndex} className="border rounded p-3 mb-3">

<div className="flex justify-between mb-2">

<span>Feature Set {groupIndex+1}</span>

{technicalIndicatorsSets.length>1 &&(

<Button
size="icon"
variant="destructive"
onClick={()=>removeFeatureSet(groupIndex)}
>
-
</Button>

)}

</div>

<div className="grid grid-cols-3 gap-2">

{featureOptions.map(set=>(
<label key={set} className="flex gap-2">

<input
type="checkbox"
checked={setGroup.includes(set)}
onChange={()=>toggleTechnicalSet(groupIndex,set)}
/>

{set}

</label>
))}

</div>

</div>

))}

<Button variant="outline" onClick={addFeatureSet}>
+ Add Feature Set
</Button>

</div>



<div>

<Label className="mb-3 block">Preprocessing</Label>

<Select value={preprocessing} onValueChange={setPreprocessing}>

<SelectTrigger>
<SelectValue/>
</SelectTrigger>

<SelectContent>

{preprocessingMethods.map(p=>(
<SelectItem key={p} value={p}>{p}</SelectItem>
))}

</SelectContent>

</Select>

</div>



<div>

<Label className="mb-3 block">Output Folder</Label>

{outputFolders.map(f=>(
<label key={f} className="flex gap-2">

<input
type="radio"
checked={outputFolder===f}
onChange={()=>setOutputFolder(f)}
/>

{f}

</label>
))}

</div>


<Button className="w-full" onClick={processData}>
🚀 Process Data
</Button>

</CardContent>

</Card>

</TabsContent>



{/* ================= PROCESS LSTM ================= */}

<TabsContent value="lstm">

<Card>

<CardHeader>
<CardTitle>Prepare LSTM Dataset</CardTitle>
</CardHeader>

<CardContent className="space-y-6">

{/* DATASET */}

<div>

<Label className="mb-3 block">Dataset</Label>

<FileSelect
filesKey="raw_data"
value={lstmDataset}
onChange={setLstmDataset}
/>

</div>


{/* STEPS */}

<div>

<Label className="mb-3 block">Processing Steps</Label>

{lstmSteps.map((s,i)=>(
<ListItem
key={s}
item={s}
onRemove={()=>setLstmSteps(lstmSteps.filter((_,x)=>x!==i))}
onMoveUp={()=>{

const arr=[...lstmSteps]
if(i===0) return
[arr[i],arr[i-1]]=[arr[i-1],arr[i]]
setLstmSteps(arr)

}}
onMoveDown={()=>{

const arr=[...lstmSteps]
if(i===arr.length-1) return
[arr[i],arr[i+1]]=[arr[i+1],arr[i]]
setLstmSteps(arr)

}}
isFirst={i===0}
isLast={i===lstmSteps.length-1}
/>
))}

</div>


{/* FEATURE SETS */}

<div>

<Label className="mb-3 block">Feature Sets</Label>

{lstmFeatureSets.map((setGroup,groupIndex)=>(

<div key={groupIndex} className="border rounded p-3 mb-3">

<div className="flex justify-between mb-2">

<span>Feature Set {groupIndex+1}</span>

{lstmFeatureSets.length>1 &&(

<Button
size="icon"
variant="destructive"
onClick={()=>removeLSTMFeatureSet(groupIndex)}
>
-
</Button>

)}

</div>

<div className="grid grid-cols-3 gap-2">

{featureOptions.map(set=>(
<label key={set} className="flex gap-2">

<input
type="checkbox"
checked={setGroup.includes(set)}
onChange={()=>toggleLSTMTechnicalSet(groupIndex,set)}
/>

{set}

</label>
))}

</div>

</div>

))}

{/* <Button variant="outline" onClick={addLSTMFeatureSet}>
+ Add Feature Set
</Button> */}

</div>


{/* PREPROCESSING */}

<div>

<Label className="mb-3 block">Preprocessing</Label>

<Select value={lstmPreprocessing} onValueChange={setLstmPreprocessing}>

<SelectTrigger>
<SelectValue/>
</SelectTrigger>

<SelectContent>

{preprocessingMethods.map(p=>(
<SelectItem key={p} value={p}>{p}</SelectItem>
))}

</SelectContent>

</Select>

</div>


{/* OUTPUT FOLDER */}

<div>

<Label className="mb-3 block">Output Folder</Label>

{outputFoldersLstm.map(f=>(
<label key={f} className="flex gap-2">

<input
type="radio"
checked={lstmOutputFolder===f}
onChange={()=>setLstmOutputFolder(f)}
/>

{f}

</label>
))}

</div>


<Button className="w-full" onClick={processLSTM}>
Generate LSTM Dataset
</Button>

</CardContent>

</Card>

</TabsContent>

{/* ================= TUNE ML ================= */}

<TabsContent value="tune">

<Card>

<CardHeader>
<CardTitle>Optuna ML Tuning</CardTitle>
</CardHeader>

<CardContent className="space-y-6">

<div>

<Label>Datasets</Label>

<Input
value={tuneFiles}
onChange={e=>setTuneFiles(e.target.value)}
placeholder="dataset1.csv,dataset2.csv"
/>

<p className="text-xs text-gray-400">
Datasets used to test different feature combinations
</p>

</div>


<div>

<Label>Trials</Label>

<Input
type="number"
value={trials}
onChange={e=>setTrials(Number(e.target.value))}
/>

<p className="text-xs text-gray-400">
Number of Optuna optimization trials
</p>

</div>


<div>

<Label>Max Features in Combination</Label>

<Input
type="number"
value={maxFeatures}
onChange={e=>setMaxFeatures(Number(e.target.value))}
/>

<p className="text-xs text-gray-400">
Maximum number of features allowed in a tested feature set
</p>

</div>


<div>

<Label>Top Feature Sets</Label>

<Input
type="number"
value={topK}
onChange={e=>setTopK(Number(e.target.value))}
/>

<p className="text-xs text-gray-400">
Number of best performing feature sets saved
</p>

</div>


<div>

<Label>Metric</Label>

<Select value={metric} onValueChange={setMetric}>

<SelectTrigger>
<SelectValue/>
</SelectTrigger>

<SelectContent>

<SelectItem value="mse">MSE</SelectItem>
<SelectItem value="accuracy">Accuracy</SelectItem>

</SelectContent>

</Select>

<p className="text-xs text-gray-400">
Metric used to evaluate model performance
</p>

</div>


<Button className="w-full" onClick={tuneData}>
Start ML Optimization
</Button>

</CardContent>

</Card>

</TabsContent>



{/* ================= TUNE LSTM ================= */}

<TabsContent value="tune-lstm">

<Card>

<CardHeader>
<CardTitle>Optuna LSTM Tuning</CardTitle>
</CardHeader>

<CardContent className="space-y-6">

<div>

<Label>Datasets</Label>

<Input
value={tuneFiles}
onChange={e=>setTuneFiles(e.target.value)}
/>

<p className="text-xs text-gray-400">
Datasets used to train and validate LSTM models
</p>

</div>


<div>

<Label>Trials</Label>

<Input
type="number"
value={trials}
onChange={e=>setTrials(Number(e.target.value))}
/>

<p className="text-xs text-gray-400">
Number of Optuna experiments
</p>

</div>


<div>

<Label>Sequence Length</Label>

<Input
type="number"
value={seqLen}
onChange={e=>setSeqLen(Number(e.target.value))}
/>

<p className="text-xs text-gray-400">
Number of timesteps used as input for LSTM
</p>

</div>


<div>

<Label>LSTM Units</Label>

<Input
type="number"
value={lstmUnits}
onChange={e=>setLstmUnits(Number(e.target.value))}
/>

<p className="text-xs text-gray-400">
Number of neurons inside each LSTM layer
</p>

</div>


<div>

<Label>LSTM Layers</Label>

<Input
type="number"
value={lstmLayers}
onChange={e=>setLstmLayers(Number(e.target.value))}
/>

<p className="text-xs text-gray-400">
Number of stacked LSTM layers
</p>

</div>


<div>

<Label>Scaler</Label>

<Select value={scalerType} onValueChange={setScalerType}>

<SelectTrigger>
<SelectValue/>
</SelectTrigger>

<SelectContent>

<SelectItem value="standard">Standard</SelectItem>
<SelectItem value="minmax">MinMax</SelectItem>

</SelectContent>

</Select>

<p className="text-xs text-gray-400">
Normalization method applied to input features
</p>

</div>


<div>

<Label>Epochs</Label>

<Input
type="number"
value={epochs}
onChange={e=>setEpochs(Number(e.target.value))}
/>

<p className="text-xs text-gray-400">
Number of training epochs
</p>

</div>


<Button className="w-full" onClick={tuneLSTM}>
Start LSTM Optimization
</Button>

</CardContent>

</Card>

</TabsContent>

</Tabs>

</div>

)

}