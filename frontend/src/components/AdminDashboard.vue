<template>
  <v-container>
    <v-row>
      <v-col cols="12">
        <h1 class="text-h4 mb-4">管理者ダッシュボード</h1>
      </v-col>
    </v-row>

    <!-- システムメトリクス -->
    <v-row>
      <v-col cols="12" md="3">
        <v-card>
          <v-card-title>総ユーザー数</v-card-title>
          <v-card-text class="text-h4">{{ metrics.total_users }}</v-card-text>
        </v-card>
      </v-col>
      <v-col cols="12" md="3">
        <v-card>
          <v-card-title>アクティブユーザー</v-card-title>
          <v-card-text class="text-h4">{{ metrics.active_users }}</v-card-text>
        </v-card>
      </v-col>
      <v-col cols="12" md="3">
        <v-card>
          <v-card-title>総チャット数</v-card-title>
          <v-card-text class="text-h4">{{ metrics.total_chats }}</v-card-text>
        </v-card>
      </v-col>
      <v-col cols="12" md="3">
        <v-card>
          <v-card-title>総メッセージ数</v-card-title>
          <v-card-text class="text-h4">{{ metrics.total_messages }}</v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <!-- ユーザー管理 -->
    <v-row class="mt-4">
      <v-col cols="12">
        <v-card>
          <v-card-title>ユーザー管理</v-card-title>
          <v-card-text>
            <v-data-table
              :headers="userHeaders"
              :items="users"
              :loading="loading"
              :items-per-page="10"
            >
              <template v-slot:item.is_active="{ item }">
                <v-switch
                  v-model="item.is_active"
                  @change="updateUserStatus(item)"
                ></v-switch>
              </template>
              <template v-slot:item.last_login="{ item }">
                {{ formatDate(item.last_login) }}
              </template>
            </v-data-table>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <!-- アクティビティログ -->
    <v-row class="mt-4">
      <v-col cols="12">
        <v-card>
          <v-card-title>最近のアクティビティ</v-card-title>
          <v-card-text>
            <v-timeline>
              <v-timeline-item
                v-for="activity in metrics.recent_activity"
                :key="activity.id"
                :color="getActivityColor(activity.activity_type)"
                :timestamp="formatDate(activity.created_at)"
              >
                {{ formatActivity(activity) }}
              </v-timeline-item>
            </v-timeline>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>
<script>
import axios from 'axios';

export default {
  name: 'AdminDashboard',
  data() {
    return {
      metrics: {
        total_users: 0,
        active_users: 0,
        total_chats: 0,
        total_messages: 0,
        recent_activity: []
      },
      users: [],
      loading: false,
      userHeaders: [
        { text: 'ID', value: 'id' },
        { text: 'ユーザー名', value: 'username' },
        { text: 'メールアドレス', value: 'email' },
        { text: 'アクティブ', value: 'is_active' },
        { text: '最終ログイン', value: 'last_login' },
        { text: '作成日時', value: 'created_at' }
      ]
    };
  },
  methods: {
    async fetchMetrics() {
      try {
        const response = await axios.get('http://localhost:8000/admin/metrics', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        this.metrics = response.data;
      } catch (error) {
        console.error('Error fetching metrics:', error);
      }
    },
    async fetchUsers() {
      this.loading = true;
      try {
        const response = await axios.get('http://localhost:8000/admin/users', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        this.users = response.data;
      } catch (error) {
        console.error('Error fetching users:', error);
      } finally {
        this.loading = false;
      }
    },
    async updateUserStatus(user) {
      try {
        await axios.put(
          `http://localhost:8000/admin/users/${user.id}/status`,
          { is_active: user.is_active },
          {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
          }
        );
      } catch (error) {
        console.error('Error updating user status:', error);
        // エラー時は元の状態に戻す
        user.is_active = !user.is_active;
      }
    },
    formatDate(date) {
      if (!date) return '';
      return new Date(date).toLocaleString();
    },
    getActivityColor(activityType) {
      const colors = {
        login: 'blue',
        chat: 'green',
        error: 'red'
      };
      return colors[activityType] || 'grey';
    },
    formatActivity(activity) {
      const user = this.users.find(u => u.id === activity.user_id);
      const username = user ? user.username : 'Unknown User';
      
      switch (activity.activity_type) {
        case 'login':
          return `${username} がログインしました`;
        case 'chat':
          return `${username} がチャットを開始しました`;
        case 'error':
          return `${username} でエラーが発生しました`;
        default:
          return `${username} のアクティビティ`;
      }
    }
  },
  mounted() {
    this.fetchMetrics();
    this.fetchUsers();
    // 定期的な更新
    setInterval(() => {
      this.fetchMetrics();
      this.fetchUsers();
    }, 30000); // 30秒ごとに更新
  }
};
</script>

<style scoped>
.v-timeline-item__body {
  max-width: 100%;
}
</style> 