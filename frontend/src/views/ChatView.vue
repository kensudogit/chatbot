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
        const response = await axios.get('http://localhost:8000/api/messages', {
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
      if (!this.newMessage.trim() || this.isLoading) return

      const userMessage = {
        role: 'user',
        content: this.newMessage
      }

      // メッセージを一時的に表示
      this.messages.push(userMessage)
      const tempMessage = this.newMessage
      this.newMessage = ''
      this.isLoading = true

      try {
        const token = localStorage.getItem('token')
        if (!token) {
          throw new Error('認証トークンが見つかりません')
        }

        const response = await axios.post('http://localhost:8000/api/chat', {
          messages: this.messages,
          is_secret: this.isSecretMode
        }, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })

        if (response.data && response.data.response) {
          const botMessage = {
            role: 'assistant',
            content: response.data.response
          }
          this.messages.push(botMessage)
          await this.$nextTick()
          this.scrollToBottom()
        } else {
          throw new Error('応答データが不正です')
        }
      } catch (error) {
        console.error('Error sending message:', error)
        // エラーが発生した場合、ユーザーメッセージを削除
        this.messages = this.messages.filter(msg => msg.content !== tempMessage)
        
        if (error.response) {
          switch (error.response.status) {
            case 401:
              // 認証エラーの場合、ログイン画面にリダイレクト
              localStorage.removeItem('token')
              this.$router.push('/login')
              break
            case 500:
              // サーバーエラーの場合、エラーメッセージを表示
              this.$vuetify.theme.themes.light.error = '#ff5252'
              this.$vuetify.theme.themes.dark.error = '#ff5252'
              this.$vuetify.theme.themes.light.primary = '#ff5252'
              this.$vuetify.theme.themes.dark.primary = '#ff5252'
              break
            default:
              // その他のエラーの場合
              console.error('Unexpected error:', error.response.data)
          }
        }
      } finally {
        this.isLoading = false
      }
    },
    scrollToBottom() {
      const container = this.$refs.messagesContainer
      container.scrollTop = container.scrollHeight
    },
    async logout() {
      try {
        const token = localStorage.getItem('token')
        await axios.post('http://localhost:8000/api/logout', {}, {
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
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  margin-bottom: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.message {
  margin-bottom: 16px;
  max-width: 80%;
  padding: 8px 16px;
  border-radius: 8px;
  position: relative;
  z-index: 1;
}

.message.user {
  margin-left: auto;
  background-color: #e3f2fd !important;
  color: #000 !important;
}

.message.assistant {
  margin-right: auto;
  background-color: #f5f5f5 !important;
  color: #000 !important;
}

.input-form {
  width: 100%;
  padding: 8px 16px;
  position: sticky;
  bottom: 0;
  background-color: inherit;
  z-index: 2;
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