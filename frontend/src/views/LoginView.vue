<template>
  <div class="login-container">
    <v-card class="login-card">
      <v-card-title>ログイン</v-card-title>
      <v-card-text>
        <v-form @submit.prevent="login" ref="form">
          <v-text-field
            v-model="email"
            label="メールアドレス"
            type="email"
            required
            :rules="[v => !!v || 'メールアドレスは必須です']"
          ></v-text-field>
          <v-text-field
            v-model="password"
            label="パスワード"
            type="password"
            required
            :rules="[v => !!v || 'パスワードは必須です']"
          ></v-text-field>
          <v-alert
            v-if="error"
            type="error"
            class="mt-3"
          >
            {{ error }}
          </v-alert>
        </v-form>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn
          color="primary darken-4"
          text-color="white"
          @click="login"
          :loading="loading"
        >
          ログイン
        </v-btn>
      </v-card-actions>
    </v-card>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'LoginView',
  data() {
    return {
      email: '',
      password: '',
      error: null,
      loading: false
    }
  },
  methods: {
    async login() {
      if (!this.$refs.form.validate()) return

      this.loading = true
      this.error = null

      try {
        const formData = new FormData()
        formData.append('username', this.email)
        formData.append('password', this.password)

        const response = await axios.post('http://localhost:8000/api/token', formData)
        
        localStorage.setItem('token', response.data.access_token)
        
        // ユーザー情報を取得して管理者権限を確認
        const userResponse = await axios.get('http://localhost:8000/api/profile', {
          headers: {
            'Authorization': `Bearer ${response.data.access_token}`
          }
        })
        
        // 管理者権限の状態を文字列として保存
        localStorage.setItem('isAdmin', String(Boolean(userResponse.data.is_admin)))
        
        // ログイン後のリダイレクト
        if (userResponse.data.is_admin) {
          this.$router.push('/admin')
        } else {
          this.$router.push('/')
        }
      } catch (error) {
        this.error = error.response?.data?.detail || 'ログインに失敗しました'
      } finally {
        this.loading = false
      }
    }
  }
}
</script>

<style scoped>
.login-container {
  max-width: 400px;
  margin: 40px auto;
  padding: 20px;
}

.login-card {
  padding: 20px;
}
</style> 