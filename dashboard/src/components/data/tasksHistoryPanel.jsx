"use client"

import React, { useEffect, useState, useRef } from "react"
import { Card, CardContent } from "../ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../ui/table"
import { Badge } from "../ui/badge"
import { Button } from "../ui/button"
import { Popover, PopoverTrigger, PopoverContent } from "../ui/popover"
import { format } from "date-fns"

export default function TasksHistoryPanel({action=""}) {
  const [tasks, setTasks] = useState([])
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const intervalRef = useRef(null)

  const fetchTasks = () => {
     fetch(`http://localhost:8000/tasks-history/?action=${action}`)
      .then((res) => res.json())
      .then((data) => {
        setTasks(data.Tasks)
        setLoading(false)
      })
      .catch((err) => {
        console.error(err)
        setLoading(false)
      })
  }

  useEffect(() => {
    fetchTasks()
  }, [])

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchTasks, 5000) 
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [autoRefresh])

  const getStateBadge = (state) => {
    switch (state) {
      case "success":
        return <Badge variant="success">Success</Badge>
      case "failed":
        return <Badge variant="destructive">Failed</Badge>
      case "created":
      case "in progress":
        return <Badge variant="warning">In progress</Badge>
      default:
        return <Badge>{state}</Badge>
    }
  }

  const formatJSON = (str) => {
    try {
      return JSON.stringify(eval(`(${str})`), null, 2)
    } catch {
      return str
    }
  }

  if (loading) return <div className="text-center py-10">Loading...</div>

  return (
    <div className="w-full py-6">
      <Card className="shadow-lg rounded-2xl">
        <CardContent>
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold">Tasks History</h2>
            <label className="flex items-center space-x-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={() => setAutoRefresh(!autoRefresh)}
                className="w-4 h-4"
              />
              <span>Auto-refresh (5s)</span>
            </label>
          </div>
          <Table className="overflow-x-auto">
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>Symbol</TableHead>
                <TableHead>Interval</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Created At</TableHead>
                <TableHead>Edited At</TableHead>
                <TableHead>Details</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {tasks.map((task) => {
                let parsedRequest = {}
                try {
                  parsedRequest = eval(`(${task.request})`)
                } catch {}

                return (
                  <TableRow key={task.id} className="">
                    <TableCell className="font-medium">{task.id}</TableCell>
                    <TableCell>{parsedRequest?.action || "N/A"}</TableCell>
                    <TableCell>{parsedRequest?.symbol || "N/A"}</TableCell>
                    <TableCell>{parsedRequest?.interval || "N/A"}</TableCell>
                    <TableCell>{getStateBadge(task.state)}</TableCell>
                    <TableCell>{format(new Date(task.created_at), "yyyy-MM-dd HH:mm")}</TableCell>
                    <TableCell>{task.edited_at ? format(new Date(task.edited_at), "yyyy-MM-dd HH:mm") : "-"}</TableCell>
                    <TableCell>
                      <Popover>
                        <PopoverTrigger asChild>
                          <Button size="sm" variant="outline">View</Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-2xl mr-4 overflow-auto p-4">
                          <div className="mb-2 text-sm font-semibold text-gray-500">Request</div>
                          <pre className="text-xs  p-2 rounded-md overflow-auto">{formatJSON(task.request)}</pre>
                          <div className="mt-4 mb-2 text-sm font-semibold text-gray-500">Response</div>
                          <pre className="text-xs  p-2 rounded-md overflow-auto">{task.response ? formatJSON(task.response) : "N/A"}</pre>
                        </PopoverContent>
                      </Popover>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
