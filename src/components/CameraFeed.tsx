// src/components/CameraFeed.tsx
import React, { useEffect, useRef, useCallback } from 'react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Camera, CameraOff } from 'lucide-react';

const OFFER_URL = 'http://localhost:8000/offer';
const STATUS_URL = 'http://localhost:8000/capture/status';

interface CameraFeedProps {
  isActive: boolean;
  isTranslating: boolean;
  onToggle: () => void;
  onFrame?: (data: { detected_sign: string }) => void;
}

export function CameraFeed({ isActive, isTranslating, onToggle, onFrame }: CameraFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const startWebRTC = useCallback(async () => {
    try {
      // 웹캠 비디오 스트림 요청
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        } 
      });
      
      // 로컬 비디오에 스트림 연결
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // RTCPeerConnection 생성
      const pc = new RTCPeerConnection();
      pcRef.current = pc;

      // WebRTC 연결에 비디오 트랙 추가
      stream.getTracks().forEach(track => pc.addTrack(track, stream));

      // Offer 생성 및 설정
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // 서버에 offer 전송
      const response = await fetch(OFFER_URL, {
        method: 'POST',
        body: JSON.stringify({ sdp: pc.localDescription }),
        headers: { 'Content-Type': 'application/json' }
      });

      const answer = await response.json();
      await pc.setRemoteDescription(new RTCSessionDescription(answer.sdp));

      console.log('WebRTC connection established');
    } catch (err) {
      console.error('Failed to start WebRTC:', err);
    }
  }, []);

  const stopWebRTC = useCallback(() => {
    // WebRTC 연결 종료
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }

    // 비디오 스트림 중지
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  // isActive 변화에 따라 WebRTC 제어
  useEffect(() => {
    if (isActive) startWebRTC();
    else stopWebRTC();

    return () => {
      stopWebRTC();
    };
  }, [isActive, startWebRTC, stopWebRTC]);

  // 번역 모드: 주기적으로 status 폴링
  useEffect(() => {
    if (!isTranslating || !onFrame) {
      if (pollTimer.current) {
        clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
      return;
    }

    const poll = async () => {
      try {
        const res = await fetch(STATUS_URL);
        if (!res.ok) return;
        const data = await res.json();
        const sign: string = data?.latest_sign ?? '';
        if (sign) onFrame({ detected_sign: sign });
      } catch (err) {
        console.error('Status fetch error:', err);
      }
    };

    poll(); // 즉시 한 번 호출
    pollTimer.current = setInterval(poll, 2000);

    return () => {
      if (pollTimer.current) {
        clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [isTranslating, onFrame]);

  return (
    <Card className="relative overflow-hidden shadow-lg bg-white border border-gray-200">
      <div className="aspect-video bg-gray-900 flex items-center justify-center relative">
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted
          className="w-full h-full object-contain" 
        />

        {!isActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white bg-black bg-opacity-50">
            <Camera size={48} className="mb-4" />
            <p>카메라가 꺼져 있습니다.</p>
          </div>
        )}

        <div className="absolute top-4 left-4">
          <Button onClick={onToggle} variant="secondary">
            {isActive ? <CameraOff className="w-4 h-4 mr-2" /> : <Camera className="w-4 h-4 mr-2" />}
            {isActive ? '카메라 중지' : '카메라 시작'}
          </Button>
        </div>
      </div>
    </Card>
  );
}