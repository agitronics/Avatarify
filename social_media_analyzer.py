import asyncio
import aiohttp
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import cv2
from bs4 import BeautifulSoup
import youtube_dl
import instaloader
from facebook_scraper import get_posts
from facenet_pytorch import MTCNN

class ImprovedSocialMediaAnalyzer(QObject):
    progress_updated = pyqtSignal(int, str)
    content_gathered = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.session = None

    async def create_session(self):
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()

    async def analyze_profile(self, url):
        await self.create_session()
        if 'facebook.com' in url:
            await self.analyze_facebook(url)
        elif 'instagram.com' in url:
            await self.analyze_instagram(url)
        elif 'youtube.com' in url:
            await self.analyze_youtube(url)
        else:
            raise ValueError("Unsupported social media platform")
        await self.close_session()

    async def analyze_facebook(self, url):
        posts = get_posts(url, pages=10)
        total_posts = 10
        for i, post in enumerate(posts):
            if 'image' in post:
                image_url = post['image']
                image = await self.download_image(image_url)
                if image is not None:
                    faces = self.detect_faces(image)
                    for face in faces:
                        self.content_gathered.emit(face)
            self.progress_updated.emit(int((i + 1) / total_posts * 100), f"Analyzed {i + 1} Facebook posts")

    async def analyze_instagram(self, url):
        L = instaloader.Instaloader()
        profile = instaloader.Profile.from_username(L.context, url.split('/')[-1])
        total_posts = min(profile.mediacount, 50)
        for i, post in enumerate(profile.get_posts()):
            if i >= total_posts:
                break
            if post.is_video:
                video = await self.download_video(post.video_url)
                faces = await self.extract_faces_from_video(video)
                for face in faces:
                    self.content_gathered.emit(face)
            else:
                image = await self.download_image(post.url)
                if image is not None:
                    faces = self.detect_faces(image)
                    for face in faces:
                        self.content_gathered.emit(face)
            self.progress_updated.emit(int((i + 1) / total_posts * 100), f"Analyzed {i + 1} Instagram posts")

    async def analyze_youtube(self, url):
        ydl_opts = {'outtmpl': 'temp_video.%(ext)s'}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        
        faces = await self.extract_faces_from_video(filename)
        for face in faces:
            self.content_gathered.emit(face)

    async def download_image(self, url):
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                image = np.asarray(bytearray(data), dtype="uint8")
                return cv2.imdecode(image, cv2.IMREAD_COLOR)
        return None

    async def download_video(self, url):
        async with self.session.get(url) as response:
            if response.status == 200:
                with open('temp_video.mp4', 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024*1024)
                        if not chunk:
                            break
                        f.write(chunk)
                return 'temp_video.mp4'
        return None

    def detect_faces(self, image):
        boxes, _ = self.mtcnn.detect(image)
        faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = image[y1:y2, x1:x2]
                faces.append(face)
        return faces

    async def extract_faces_from_video(self, video_path):
        faces = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total_frames):
            ret, frame = cap.read()
            if ret:
                faces.extend(self.detect_faces(frame))
            if i % 30 == 0:
                self.progress_updated.emit(int(i / total_frames * 100), f"Processed {i} frames from video")
        cap.release()
        return faces