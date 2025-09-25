import React, { useState } from "react"
import { motion, AnimatePresence } from "motion/react"
import { Link } from "react-router-dom"
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
import { Home, Inbox, Calendar, Search, Settings, User2, GraduationCap } from "lucide-react"
import { ChevronDown, ChevronUp } from "lucide-react"

function LeftSidebar({ children }) {
  const [openMenus, setOpenMenus] = useState({})

  const toggleMenu = (title) => setOpenMenus(prev => ({ ...prev, [title]: !prev[title] }))

  const menuItems = [
    { title: "Dashboard", icon: Home, submenu: ["Home", "Dashboard"] },
    { title: "Data", icon: Inbox, submenu: ["Historical data", "Process data"] },
    { title: "Machine learning", icon: Calendar, submenu: ["Training", "Models"] },
    { title: "Reinforcement learning", icon: Search, submenu: ["Learning", "Models"] },
    { title: "Tests", icon: Settings, submenu: ["Tests", "Simulation", "History"] },
    { title: "Explainable AI", icon: GraduationCap, submenu: ["Explain", "History"] },
  ]

  const submenuVariants = { hidden: { height: 0, opacity: 0, overflow: "hidden" }, visible: { height: "auto", opacity: 1, overflow: "hidden" } }

  const slugify = (str) => str.toLowerCase().replace(/\s+/g, "-")

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full">
        <Sidebar>
          <SidebarHeader>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton onClick={() => toggleMenu("Workspace")}>
                  Workspace {openMenus["Workspace"] ? <ChevronUp className="ml-auto" /> : <ChevronDown className="ml-auto" />}
                </SidebarMenuButton>
                <AnimatePresence>
                  {openMenus["Workspace"] && (
                    <motion.div initial="hidden" animate="visible" exit="hidden" variants={submenuVariants} className="ml-4 mt-2 flex flex-col gap-1">
                      <Link className="px-4 py-2 hover:bg-slate-700 rounded" to="/acme-inc">Acme Inc</Link>
                      <Link className="px-4 py-2 hover:bg-slate-700 rounded" to="/acme-corp">Acme Corp</Link>
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
                  {menuItems.map(item => (
                    <SidebarMenuItem key={item.title} className="my-1">
                      <SidebarMenuButton type="button" className="flex items-center gap-2 cursor-pointer" onClick={() => toggleMenu(item.title)}>
                        <item.icon /> <span>{item.title}</span>
                        {item.submenu?.length > 0 && (openMenus[item.title] ? <ChevronUp className="ml-auto"/> : <ChevronDown className="ml-auto"/>)}
                      </SidebarMenuButton>

                      <AnimatePresence>
                        {item.submenu && openMenus[item.title] && (
                          <motion.div initial="hidden" animate="visible" exit="hidden" variants={submenuVariants} transition={{duration:0.2}} className="ml-6 mt-2 px-2 flex flex-col gap-1">
                            {item.submenu.map(subItem => (
                              <Link key={subItem} to={`/${slugify(subItem)}`} className="px-4 py-2 hover:bg-slate-700 rounded">
                                {subItem}
                              </Link>
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
                  <User2 /> Username {openMenus["User"] ? <ChevronUp className="ml-auto" /> : <ChevronDown className="ml-auto" />}
                </SidebarMenuButton>
                <AnimatePresence>
                  {openMenus["User"] && (
                    <motion.div initial="hidden" animate="visible" exit="hidden" variants={submenuVariants} className="ml-4 mt-2 flex flex-col gap-1">
                      <Link className="px-4 py-2 hover:bg-slate-700 rounded" to="/account">Account</Link>
                      <Link className="px-4 py-2 hover:bg-slate-700 rounded" to="/billing">Billing</Link>
                      <Link className="px-4 py-2 hover:bg-slate-700 rounded" to="/sign-out">Sign out</Link>
                    </motion.div>
                  )}
                </AnimatePresence>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarFooter>
        </Sidebar>

        <main className="w-full flex-1 px-6 py-2 overflow-y-auto">
          {children}
        </main>
      </div>
    </SidebarProvider>
  )
}

export default LeftSidebar
