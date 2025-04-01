<!--
アプリケーションのルートコンポーネント
- アプリケーションのレイアウト
- テーマ設定
- ルーティング
-->

<template>
  <v-app>
    <!-- アプリケーションバー -->
    <v-app-bar app>
      <v-toolbar-title>AIチャットボット</v-toolbar-title>
      <v-spacer></v-spacer>
      <!-- テーマ切り替えボタン -->
      <v-btn icon @click="toggleTheme">
        <v-icon>{{ isDark ? 'mdi-weather-sunny' : 'mdi-weather-night' }}</v-icon>
      </v-btn>
    </v-app-bar>

    <!-- メインコンテンツ -->
    <v-main>
      <v-container>
        <router-view></router-view>
      </v-container>
    </v-main>
  </v-app>
</template>

<script>
import { ref, watch } from 'vue'
import { useTheme } from 'vuetify'

export default {
  name: 'App',
  
  setup() {
    const theme = useTheme()
    const isDark = ref(false)

    // テーマを切り替える関数
    const toggleTheme = () => {
      isDark.value = !isDark.value
      theme.global.name.value = isDark.value ? 'dark' : 'light'
    }

    // システムのテーマ設定を監視
    watch(
      () => window.matchMedia('(prefers-color-scheme: dark)').matches,
      (isSystemDark) => {
        isDark.value = isSystemDark
        theme.global.name.value = isSystemDark ? 'dark' : 'light'
      },
      { immediate: true }
    )

    return {
      isDark,
      toggleTheme
    }
  }
}
</script>

<style>
/* アプリケーション全体のスタイル */
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  font-family: 'Roboto', sans-serif;
}

/* ダークモード対応 */
.v-theme--dark {
  background-color: #121212;
  color: #ffffff;
}
</style> 