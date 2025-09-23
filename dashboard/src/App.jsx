"use client"

import { useEffect, useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { motion } from "motion/react"
import { FcGoogle } from "react-icons/fc"
import { toast, Toaster } from "sonner"
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom"
import HomePage from "./pages/HomePage.jsx"
import SimulationPage from "./pages/tests/SimulationPage.jsx"

const MotionButton = motion(Button)

export default function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const hasFetched = useRef(false)

  const handleLogin = () => {
    window.location.href = "http://localhost:8000/auth/login"
  }

  const checkLoginResult = async () => {
    try {
      const response = await fetch("http://localhost:8000/auth/me", {
        method: "GET",
        credentials: "include",
      })
      if (response.ok) {
        const data = await response.json()
        setUser({ id: data.id, name: data.name })
        toast.success(`Welcome, ${data.name}!`, { position: "top-right" })
      } else {
        setUser(null)
        toast.error("User is not logged in", { position: "top-right" })
      }
    } catch (err) {
      console.error("Error checking login:", err)
      toast.error("Error checking login", { position: "top-right" })
      setUser(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (hasFetched.current) return
    hasFetched.current = true
    checkLoginResult()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
          className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto"
        />
      </div>
    )
  }

  return (
    <Router>
      <Routes>
        {/* /home dostępne tylko dla zalogowanego */}
        <Route path="/home" element={user ? <HomePage /> : <Navigate to="/" replace />} />
        <Route path="/simulation" element={user ? <SimulationPage /> : <Navigate to="/" replace />} />

        {/* Strona główna */}
        <Route
          path="/"
          element={
            user ? (
              <Navigate to="/home" replace />
            ) : (
              <div className="flex items-center justify-center min-h-screen">
                <motion.div
                  initial={{ opacity: 0.2, y: -50 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="p-6 rounded-2xl shadow-md text-center"
                >
                  <h1 className="text-4xl font-bold mb-12">Welcome</h1>
                  <MotionButton
                    onClick={handleLogin}
                    variant="outline"
                    className="flex items-center gap-2"
                  >
                    <FcGoogle size={20} />
                    Login with Google
                  </MotionButton>
                </motion.div>
                <Toaster position="top-right" />
              </div>
            )
          }
        />
      </Routes>
    </Router>
  )
}
