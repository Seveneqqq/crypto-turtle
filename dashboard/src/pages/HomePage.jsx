"use client"

import React, { useState } from "react"
import { motion, AnimatePresence } from "motion/react"
import {
  SidebarProvider,
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "../components/ui/sidebar"

import ApiPanel from "../components/data/apiPanel.jsx"
import TasksHistoryPanel from "../components/data/tasksHistoryPanel.jsx"

import { Home, Inbox, Calendar, Search, Settings, User2, GraduationCap } from "lucide-react"
import { ChevronDown, ChevronUp } from "lucide-react"
import FilePanel from "../components/data/filePanel.jsx"
import SimulatePanel from "../components/simulation/simulatePanel.jsx"

export default function HomePage() {
  const [openMenus, setOpenMenus] = useState({})

  const toggleMenu = (title) => {
    setOpenMenus((prev) => ({ ...prev, [title]: !prev[title] }))
  }

  const menuItems = [
    { title: "Dashboard", icon: Home, submenu: ["Dashboard"] },
    { title: "Data", icon: Inbox, submenu: ["Data", "Processing"] },
    { title: "Machine learning", icon: Calendar, submenu: ["Training", "Models"] },
    { title: "Reinforcement learning", icon: Search, submenu: ["Learnig", "Models"] },
    { title: "Tests", icon: Settings, submenu: ["Tests", "History"] },
    { title: "Explainable AI", icon: GraduationCap, submenu: ["Explain", "History"] },
  ]

  const submenuVariants = {
    hidden: { height: 0, opacity: 0, overflow: "hidden" },
    visible: { height: "auto", opacity: 1, overflow: "hidden" },
  }

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full">
        <Sidebar>
          <SidebarHeader>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton onClick={() => toggleMenu("Workspace")}>
                  Workspace
                  {openMenus["Workspace"] ? (
                    <ChevronUp className="ml-auto" />
                  ) : (
                    <ChevronDown className="ml-auto" />
                  )}
                </SidebarMenuButton>
                <AnimatePresence>
                  {openMenus["Workspace"] && (
                    <motion.div
                      initial="hidden"
                      animate="visible"
                      exit="hidden"
                      variants={submenuVariants}
                      className="ml-4 mt-2 flex flex-col gap-1"
                    >
                      <a href="#">Acme Inc</a>
                      <a href="#">Acme Corp</a>
                    </motion.div>
                  )}
                </AnimatePresence>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarHeader>

          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupLabel>Application</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {menuItems.map((item) => (
                    <SidebarMenuItem key={item.title} className="my-1">
                      <SidebarMenuButton
                        className="flex items-center gap-2"
                        onClick={() => toggleMenu(item.title)}
                      >
                        <item.icon />
                        <span>{item.title}</span>
                        {item.submenu && item.submenu.length > 0 && (
                          openMenus[item.title] ? <ChevronUp className="ml-auto" /> : <ChevronDown className="ml-auto" />
                        )}
                      </SidebarMenuButton>

                      <AnimatePresence>
                        {item.submenu && openMenus[item.title] && (
                          <motion.div
                            initial="hidden"
                            animate="visible"
                            exit="hidden"
                            variants={submenuVariants}
                            transition={{ duration: 0.2 }}
                            className="ml-6 mt-2 px-2 flex flex-col gap-1"
                          >
                            {item.submenu.map((subItem) => (
                              <a
                                key={subItem}
                                href="#"
                                className="px-4 py-2 hover:bg-slate-700 rounded"
                              >
                                {subItem}
                              </a>
                            ))}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>

          <SidebarFooter>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton onClick={() => toggleMenu("User")}>
                  <User2 />
                  Username
                  {openMenus["User"] ? <ChevronUp className="ml-auto" /> : <ChevronDown className="ml-auto" />}
                </SidebarMenuButton>
                <AnimatePresence>
                  {openMenus["User"] && (
                    <motion.div
                      initial="hidden"
                      animate="visible"
                      exit="hidden"
                      variants={submenuVariants}
                      className="ml-4 mt-2 flex flex-col gap-1"
                    >
                      <a href="#">Account</a>
                      <a href="#">Billing</a>
                      <a href="#">Sign out</a>
                    </motion.div>
                  )}
                </AnimatePresence>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarFooter>
        </Sidebar>

        <main className="w-full flex-1 px-6 py-2">
          <h1 className="text-3xl font-bold my-2">Welcome to the Home Page!</h1>
          <div className="flex gap-6">
                {/* <ApiPanel />
                <FilePanel /> */}
                <SimulatePanel />
          </div>
          <TasksHistoryPanel />
        </main>
      </div>
    </SidebarProvider>
  )
}
