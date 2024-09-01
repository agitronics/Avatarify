import { useState, useEffect } from 'react'
import Head from 'next/head'
import styles from '../styles/Home.module.css'

export default function Home() {
  const [videoStream, setVideoStream] = useState(null)
  const [effects, setEffects] = useState([])
  const [selectedEffect, setSelectedEffect] = useState('')

  useEffect(() => {
    // Connect to backend websocket for real-time video streaming
    const socket = new WebSocket('ws://localhost:8000/ws')
    socket.onmessage = (event) => {
      const videoBlob = new Blob([event.data], { type: 'video/webm' })
      setVideoStream(URL.createObjectURL(videoBlob))
    }

    // Fetch available effects from the backend
    fetch('http://localhost:8000/effects')
      .then(response => response.json())
      .then(data => setEffects(data))

    return () => {
      socket.close()
    }
  }, [])

  const applyEffect = () => {
    fetch('http://localhost:8000/apply-effect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ effect: selectedEffect })
    })
  }

  return (
    <div className={styles.container}>
      <Head>
        <title>Avatarify Web Interface</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>Avatarify Web Interface</h1>

        <div className={styles.videoContainer}>
          {videoStream && <video src={videoStream} autoPlay />}
        </div>

        <div className={styles.controls}>
          <select 
            value={selectedEffect} 
            onChange={(e) => setSelectedEffect(e.target.value)}
          >
            <option value="">Select an effect</option>
            {effects.map(effect => (
              <option key={effect} value={effect}>{effect}</option>
            ))}
          </select>
          <button onClick={applyEffect}>Apply Effect</button>
        </div>
      </main>
    </div>
  )
}