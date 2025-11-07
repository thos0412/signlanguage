// src/components/ControlPanel.tsx

import React from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Label } from './ui/label';
import { Separator } from './ui/separator';
import { Play, Pause, RotateCcw, Settings } from 'lucide-react';

interface ControlPanelProps {
  isTranslating: boolean;
  onToggleTranslation: () => void;
  onClearHistory: () => void;
  tone: string; // 기존 language 대신 tone
  onToneChange: (tone: string) => void; // 기존 onLanguageChange 대신
}

export function ControlPanel({
  isTranslating,
  onToggleTranslation,
  onClearHistory,
  tone,
  onToneChange,
}: ControlPanelProps) {
  const tones = [
    { value: 'formal', label: '존댓말' },
    { value: 'informal', label: '반말' },
  ];

  return (
    <div className="space-y-6">
      <Card className="shadow-lg"
        style={{ backgroundColor: '#9bbbd4'}}>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="w-5 h-5 text-gray-700" />
            <span>Controls</span>
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Translation Control */}
          <div className="space-y-3">
            <Button 
              onClick={onToggleTranslation}
              className={`w-full ${
                isTranslating 
                  ? 'bg-red-600 hover:bg-red-700' 
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
              size="lg"
            >
              {isTranslating ? (
                <>
                  <Pause className="w-4 h-4 mr-2" />
<<<<<<< HEAD
                  수어 인식 중지
=======
                  번역 중지
>>>>>>> e31caaf17ed9e45b694eb3c04227520acaf5e330
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
<<<<<<< HEAD
                  수어 인식 시작
=======
                  번역 시작
>>>>>>> e31caaf17ed9e45b694eb3c04227520acaf5e330
                </>
              )}
            </Button>
            
            <Button 
              onClick={onClearHistory}
              variant="outline"
              className="w-full"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              초기화
            </Button>
          </div>

          <Separator />

          {/* Tone Selection */}
          <div className="space-y-2">
            <Label>어투 선택</Label>
            <Select value={tone} onValueChange={onToneChange}>
              <SelectTrigger>
                <SelectValue placeholder="어투 선택" />
              </SelectTrigger>
              <SelectContent>
                {tones.map((t) => (
                  <SelectItem key={t.value} value={t.value}>
                    {t.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Status Panel */}
      <Card className="shadow-lg"
        style={{ backgroundColor: '#9bbbd4'}}>
        <CardContent className="p-4">
          <div className="text-center space-y-2">
            <div className="flex items-center justify-center space-x-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  isTranslating ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                }`}
              ></div>
              <span className="text-sm text-black-700">
                {isTranslating ? '번역 중' : '대기 중'}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
