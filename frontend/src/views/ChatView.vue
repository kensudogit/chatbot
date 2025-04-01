<template>
  <div class="chat-container">
    <v-card class="chat-card" :dark="isDarkMode">
      <v-card-title class="d-flex justify-space-between align-center">
        <span>チャット</span>
        <div class="d-flex align-center">
          <v-tooltip location="bottom">
            <template v-slot:activator="{ props }">
              <v-btn
                icon
                :color="isDarkMode ? 'white' : (isSecretMode ? 'error' : 'grey')"
                @click="toggleDarkMode"
                v-bind="props"
                class="mr-4"
              >
                <v-icon>
                  <img :src="require('@/assets/pc.jpg')" alt="Dark Mode" style="width: 24px; height: 24px;">
                </v-icon>
              </v-btn>
            </template>
            <span>{{ isDarkMode ? 'ライトモード' : 'ダークモード' }}</span>
          </v-tooltip>
          <v-tooltip location="bottom">
            <template v-slot:activator="{ props }">
              <v-btn
                icon
                :color="isDarkMode ? 'white' : (isSecretMode ? 'error' : 'grey')"
                @click="toggleSecretMode"
                v-bind="props"
                class="mr-4"
              >
                <v-icon>
                  <img :src="require('@/assets/secret.jpg')" alt="Secret Mode" style="width: 24px; height: 24px;">
                </v-icon>
              </v-btn>
            </template>
            <span>{{ isSecretMode ? 'シークレットモード: オン' : 'シークレットモード: オフ' }}</span>
          </v-tooltip>
          <v-btn
            :color="isDarkMode ? 'white' : '#1a237c'"
            :dark="!isDarkMode"
            :light="isDarkMode"
            @click="logout"
            class="logout-btn"
            elevation="2"
          >
            ログアウト
          </v-btn>
        </div>
      </v-card-title>
      <v-card-text>
        <div class="messages" ref="messagesContainer">
          <v-card 
            v-for="(message, index) in messages" 
            :key="index" 
            :class="['message', message.role]" 
            :dark="isDarkMode"
            flat
          >
            <v-card-text>{{ message.content }}</v-card-text>
          </v-card>
        </div>
      </v-card-text>
      <v-card-actions>
        <v-form @submit.prevent="sendMessage" class="input-form">
          <v-text-field
            v-model="newMessage"
            label="メッセージを入力"
            append-icon="mdi-send"
            @click:append="sendMessage"
            @keyup.enter="sendMessage"
            :dark="isDarkMode"
          ></v-text-field>
        </v-form>
      </v-card-actions>
    </v-card>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'ChatView',
  data() {
    return {
      messages: [],
      newMessage: '',
      isSecretMode: false,
      isDarkMode: false,
      isLoading: false
    }
  },
  async created() {
    await this.loadMessages()
  },
  methods: {
    async loadMessages() {
      try {
        const token = localStorage.getItem('token')
        if (!token) {
          throw new Error('認証トークンが見つかりません')
        }

        // メッセージ取得APIを呼び出し
        const response = await axios.get('http://localhost:8080/api/messages', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })

        // メッセージを設定
        if (response.data && response.data.messages) {
          this.messages = response.data.messages
        } else {
          this.messages = []
        }
      } catch (error) {
        console.error('Error loading messages:', error)
        if (error.response) {
          switch (error.response.status) {
            case 401:
              // 認証エラーの場合、ログイン画面にリダイレクト
              localStorage.removeItem('token')
              this.$router.push('/login')
              break
            default:
              console.error('Unexpected error:', error.response.data)
          }
        }
      }
    },
    toggleDarkMode() {
      this.isDarkMode = !this.isDarkMode
      this.$vuetify.theme.dark = this.isDarkMode
    },
    toggleSecretMode() {
      this.isSecretMode = !this.isSecretMode
      if (this.isSecretMode) {
        this.$vuetify.theme.themes.light.primary = '#ff5252'
        this.$vuetify.theme.themes.dark.primary = '#ff5252'
      } else {
        this.$vuetify.theme.themes.light.primary = '#1976d2'
        this.$vuetify.theme.themes.dark.primary = '#1976d2'
      }
    },
    async sendMessage() {
      if (!this.newMessage.trim()) return

      const userMessage = {
        role: 'user',
        content: this.newMessage
      }

      // 一時的にメッセージを保存
      const currentMessage = this.newMessage
      this.messages.push(userMessage)
      this.newMessage = ''

      try {
        const response = await axios.post('http://localhost:8080/api/chat', {
          messages: [userMessage]
        })

        if (response.data && response.data.response) {
          const botMessage = {
            role: 'assistant',
            content: response.data.response
          }
          this.messages.push(botMessage)
        }
      } catch (error) {
        console.error('Error:', error)
        // エラー時は送信したメッセージを削除
        this.messages = this.messages.filter(msg => msg.content !== currentMessage)
      } finally {
        // メッセージの追加後に必ずスクロールを実行
        await this.$nextTick()
        this.scrollToBottom()
      }
    },
    scrollToBottom() {
      const container = this.$refs.messagesContainer
      if (container) {
        // スムーズスクロールを使用
        container.scrollTo({
          top: container.scrollHeight,
          behavior: 'smooth'
        })
      }
    },
    async logout() {
      try {
        const token = localStorage.getItem('token')
        await axios.post('http://localhost:8080/api/logout', {}, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })
        localStorage.removeItem('token')
        this.$router.push('/login')
      } catch (error) {
        console.error('Logout error:', error)
      }
    }
  }
}
</script>

<style scoped>
.chat-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  height: 100vh;
}

.chat-card {
  height: calc(100vh - 40px);
  display: flex;
  flex-direction: column;
}

.v-card-text {
  flex: 1;
  padding: 0;
  position: relative;
  height: calc(100vh - 200px);
  overflow: hidden;
}

.messages {
  height: 100%;
  overflow-y: auto !important;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  background-color: rgba(0, 0, 0, 0.02);
}

.message {
  margin-bottom: 10px;
  padding: 15px;
  border-radius: 8px;
  max-width: 80%;
  word-break: break-word;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.message.user {
  align-self: flex-end;
  background-color: #1976d2;
  color: white;
}

.message.assistant {
  align-self: flex-start;
  background-color: #ffffff;
  color: black;
  border: 1px solid rgba(0, 0, 0, 0.1);
}

.input-form {
  width: 100%;
  padding: 16px;
  background-color: transparent;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

/* ダークモード時のスタイル */
.v-theme--dark .message.assistant {
  background-color: #424242;
  color: white;
  border-color: rgba(255, 255, 255, 0.1);
}

.v-theme--dark .messages {
  background-color: rgba(255, 255, 255, 0.02);
}

/* スクロールバーのスタイル */
.messages::-webkit-scrollbar {
  width: 10px;
}

.messages::-webkit-scrollbar-track {
  background: transparent;
}

.messages::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 10px;
  border: 2px solid transparent;
  background-clip: padding-box;
}

.messages::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.3);
  border: 2px solid transparent;
  background-clip: padding-box;
}

/* Firefox用のスクロールバースタイル */
.messages {
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 0, 0, 0.2) transparent;
}

/* ダークモード時のスクロールバー */
.v-theme--dark .messages::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
}

.v-theme--dark .messages::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.3);
}

.v-theme--dark .messages {
  scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
}

.v-card-title {
  display: flex;
  justify-content: flex-start;
  align-items: center;
  padding: 16px;
  width: 100%;
  position: sticky;
  top: 0;
  z-index: 2;
  background-color: inherit;
}

.v-card-title span {
  margin-right: auto;
  font-size: 20px;
  font-weight: 500;
}

.d-flex {
  display: flex;
  align-items: center;
  margin-left: auto;
}

.logout-btn {
  margin-left: 0;
  margin-right: 16px;
  text-transform: none !important;
  letter-spacing: 0.4px;
  font-weight: 500;
  font-size: 12px;
  padding: 0 10px !important;
  height: 30px !important;
}

.v-btn--icon {
  transition: all 0.3s ease;
}

.v-btn--icon:hover {
  transform: scale(1.1);
}

.v-card-actions {
  padding: 0 16px 16px 16px;
  position: sticky;
  bottom: 0;
  background-color: inherit;
  z-index: 2;
}

.v-text-field {
  width: 100%;
}
</style> 