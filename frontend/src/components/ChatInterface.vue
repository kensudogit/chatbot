<template>
  <v-container class="chat-container">
    <v-card class="chat-card">
      <v-card-title class="chat-title">
        チャットボット
      </v-card-title>
      
      <v-card-text class="messages-container" ref="messagesContainer">
        <div v-for="(message, index) in messages" :key="index" 
             :class="['message', message.role === 'user' ? 'user-message' : 'bot-message']">
          <div class="message-content">
            <template v-if="message.role === 'user'">
              {{ message.content }}
            </template>
            <template v-else>
              <span v-for="(char, charIndex) in message.displayedContent" :key="charIndex">{{ char }}</span>
            </template>
          </div>
          <div v-if="message.sentiment" class="sentiment-indicator">
            感情分析: {{ message.sentiment.label }} ({{ (message.sentiment.score * 100).toFixed(1) }}%)
          </div>
        </div>
      </v-card-text>

      <v-card-actions class="input-container">
        <v-file-input
          v-model="audioFile"
          accept="audio/*"
          label="音声ファイル"
          prepend-icon="mdi-microphone"
          @change="handleAudioUpload"
          hide-details
          class="audio-input"
        ></v-file-input>
        
        <v-text-field
          v-model="userInput"
          label="メッセージを入力"
          @keyup.enter="sendMessage"
          hide-details
          class="message-input"
        ></v-text-field>
        
        <v-btn
          color="primary"
          @click="sendMessage"
          :loading="loading"
        >
          送信
        </v-btn>
        
        <v-btn
          color="secondary"
          @click="startRecording"
          :disabled="isRecording"
        >
          <v-icon>mdi-microphone</v-icon>
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script>
import axios from 'axios';

export default {
  name: 'ChatInterface',
  data() {
    return {
      messages: [],
      userInput: '',
      loading: false,
      audioFile: null,
      isRecording: false,
      mediaRecorder: null,
      audioChunks: [],
      typingSpeed: 50, // タイピング速度（ミリ秒）
    };
  },
  methods: {
    async sendMessage() {
      if (!this.userInput.trim()) return;

      // ユーザーメッセージを追加
      this.messages.push({
        role: 'user',
        content: this.userInput
      });

      const userMessage = this.userInput;
      this.userInput = '';
      this.loading = true;

      try {
        // APIリクエスト
        const response = await axios.post('http://localhost:8000/chat', {
          messages: this.messages
        }, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });

        // ボットの応答を追加（表示用の配列を初期化）
        this.messages.push({
          role: 'assistant',
          content: response.data.response,
          displayedContent: [],
          sentiment: response.data.sentiment
        });

        // 逐次的に表示
        await this.typeMessage(this.messages[this.messages.length - 1]);

        // 音声出力
        await this.textToSpeech(response.data.response);
      } catch (error) {
        console.error('Error:', error);
        this.messages.push({
          role: 'assistant',
          content: '申し訳ありません。エラーが発生しました。',
          displayedContent: []
        });
        await this.typeMessage(this.messages[this.messages.length - 1]);
      } finally {
        this.loading = false;
        this.$nextTick(() => {
          this.scrollToBottom();
        });
      }
    },

    async typeMessage(message) {
      const chars = message.content.split('');
      for (let char of chars) {
        message.displayedContent.push(char);
        await new Promise(resolve => setTimeout(resolve, this.typingSpeed));
        this.$nextTick(() => {
          this.scrollToBottom();
        });
      }
    },

    async handleAudioUpload(file) {
      if (!file) return;

      const formData = new FormData();
      formData.append('audio_file', file);

      try {
        const response = await axios.post('http://localhost:8000/speech-to-text', formData, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });

        this.userInput = response.data.text;
        await this.sendMessage();
      } catch (error) {
        console.error('Error:', error);
      }
    },

    async startRecording() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.mediaRecorder = new MediaRecorder(stream);
        this.audioChunks = [];

        this.mediaRecorder.ondataavailable = (event) => {
          this.audioChunks.push(event.data);
        };

        this.mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
          const formData = new FormData();
          formData.append('audio_file', audioBlob);

          try {
            const response = await axios.post('http://localhost:8000/speech-to-text', formData, {
              headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
              }
            });

            this.userInput = response.data.text;
            await this.sendMessage();
          } catch (error) {
            console.error('Error:', error);
          }
        };

        this.mediaRecorder.start();
        this.isRecording = true;

        // 5秒後に録音を停止
        setTimeout(() => {
          this.stopRecording();
        }, 5000);
      } catch (error) {
        console.error('Error:', error);
      }
    },

    stopRecording() {
      if (this.mediaRecorder && this.isRecording) {
        this.mediaRecorder.stop();
        this.isRecording = false;
      }
    },

    async textToSpeech(text) {
      try {
        const response = await axios.post('http://localhost:8000/text-to-speech', 
          { text },
          {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            responseType: 'blob'
          }
        );

        const audio = new Audio(URL.createObjectURL(response.data));
        await audio.play();
      } catch (error) {
        console.error('Error:', error);
      }
    },

    scrollToBottom() {
      const container = this.$refs.messagesContainer;
      container.scrollTop = container.scrollHeight;
    }
  }
};
</script>

<style scoped>
.chat-container {
  height: 100vh;
  padding: 20px;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
}

.chat-card {
  height: 100%;
  display: flex;
  flex-direction: column;
  position: relative;
  border-radius: 8px;
  overflow: hidden;
}

.chat-title {
  background-color: #1976D2;
  color: white;
  padding: 10px;
  font-size: 1.2em;
  text-align: center;
  flex-shrink: 0;
}

.messages-container {
  position: relative;
  height: calc(100vh - 280px);
  overflow-y: scroll;
  overflow-x: hidden;
  padding: 20px;
  padding-bottom: 140px;
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
  box-sizing: border-box;
  will-change: transform;
}

.messages-container::-webkit-scrollbar {
  width: 8px;
}

.messages-container::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.messages-container::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 4px;
}

.messages-container::-webkit-scrollbar-thumb:hover {
  background: #555;
}

.message {
  margin-bottom: 16px;
  animation: fadeIn 0.3s ease-in-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.user-message {
  background-color: #E3F2FD;
  margin-left: auto;
}

.bot-message {
  background-color: #F5F5F5;
  margin-right: auto;
}

.message-content {
  margin-bottom: 5px;
}

.sentiment-indicator {
  font-size: 0.8em;
  color: #666;
  margin-top: 5px;
}

.input-container {
  position: fixed;
  bottom: 20px;
  left: 20px;
  right: 20px;
  min-height: 80px;
  height: auto;
  max-height: 120px;
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
  z-index: 1000;
  display: flex;
  gap: 10px;
  align-items: center;
  transform: translateZ(0);
  -webkit-transform: translateZ(0);
  margin-bottom: 0;
  box-sizing: border-box;
}

.v-text-field {
  height: auto !important;
  min-height: 40px !important;
  max-height: 80px !important;
  flex: 1 !important;
  margin: 0 !important;
  padding: 0 !important;
}

.v-text-field.v-text-field--enclosed .v-text-field__details {
  display: none !important;
}

.v-text-field.v-text-field--enclosed .v-input__slot {
  min-height: 40px !important;
  padding: 0 12px !important;
}

.v-text-field textarea {
  min-height: 40px !important;
  max-height: 80px !important;
  line-height: 1.5 !important;
  padding: 8px 0 !important;
}

.audio-input {
  width: 180px !important;
  flex: 0 0 auto !important;
  margin: 0 !important;
  height: 40px !important;
}

.v-btn {
  height: 40px !important;
  width: 40px !important;
  min-width: 40px !important;
  flex: 0 0 auto !important;
  margin: 0 !important;
  padding: 0 !important;
}

.send-button {
  background-color: #1976D2;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
}

.send-button:hover {
  background-color: #1565C0;
}

.record-button {
  background-color: #4CAF50;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
}

.record-button:hover {
  background-color: #388E3C;
}

.record-button.recording {
  background-color: #f44336;
}

.record-button.recording:hover {
  background-color: #d32f2f;
}
</style> 