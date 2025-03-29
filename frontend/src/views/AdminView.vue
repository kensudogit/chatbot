<template>
  <div class="admin-container">
    <v-card>
      <v-card-title class="headline">管理者ダッシュボード</v-card-title>
      <v-card-text>
        <v-row>
          <v-col cols="12" md="6">
            <v-card>
              <v-card-title>システムメトリクス</v-card-title>
              <v-card-text v-if="metrics">
                <v-list>
                  <v-list-item>
                    <template v-slot:title>
                      総ユーザー数
                    </template>
                    <template v-slot:subtitle>
                      {{ metrics.total_users || 0 }}
                    </template>
                  </v-list-item>
                  <v-list-item>
                    <template v-slot:title>
                      アクティブユーザー
                    </template>
                    <template v-slot:subtitle>
                      {{ metrics.active_users || 0 }}
                    </template>
                  </v-list-item>
                  <v-list-item>
                    <template v-slot:title>
                      総チャット数
                    </template>
                    <template v-slot:subtitle>
                      {{ metrics.total_chats || 0 }}
                    </template>
                  </v-list-item>
                  <v-list-item>
                    <template v-slot:title>
                      総メッセージ数
                    </template>
                    <template v-slot:subtitle>
                      {{ metrics.total_messages || 0 }}
                    </template>
                  </v-list-item>
                  <v-list-item>
                    <template v-slot:title>
                      平均応答時間
                    </template>
                    <template v-slot:subtitle>
                      {{ metrics.avg_response_time || 0 }}ms
                    </template>
                  </v-list-item>
                </v-list>
              </v-card-text>
              <v-card-text v-else>
                <v-progress-circular indeterminate color="primary"></v-progress-circular>
              </v-card-text>
            </v-card>
          </v-col>
          <v-col cols="12" md="6">
            <v-card>
              <v-card-title>最近のアクティビティ</v-card-title>
              <v-card-text v-if="metrics && metrics.recent_activity">
                <v-timeline>
                  <v-timeline-item
                    v-for="activity in metrics.recent_activity"
                    :key="activity.id"
                    :dot-color="getActivityColor(activity.activity_type)"
                    size="small"
                  >
                    <template v-slot:opposite>
                      <div class="text-caption">{{ formatDate(activity.created_at) }}</div>
                    </template>
                    <v-card>
                      <v-card-text>
                        <div class="text-subtitle-1">{{ activity.activity_type }}</div>
                        <div class="text-caption">{{ activity.description }}</div>
                      </v-card-text>
                    </v-card>
                  </v-timeline-item>
                </v-timeline>
              </v-card-text>
              <v-card-text v-else-if="!metrics">
                <v-progress-circular indeterminate color="primary"></v-progress-circular>
              </v-card-text>
              <v-card-text v-else>
                <v-alert type="info" text>
                  最近のアクティビティはありません
                </v-alert>
              </v-card-text>
            </v-card>
          </v-col>
        </v-row>
        <v-row class="mt-4">
          <v-col cols="12">
            <v-card>
              <v-card-title>システム状態</v-card-title>
              <v-card-text v-if="metrics">
                <v-chip
                  :color="metrics.system_status === 'healthy' ? 'success' : 'error'"
                  class="mr-2"
                >
                  {{ metrics.system_status === 'healthy' ? '正常' : '異常' }}
                </v-chip>
                <v-chip
                  :color="getMemoryUsageColor(metrics.memory_usage)"
                  class="mr-2"
                >
                  メモリ使用率: {{ metrics.memory_usage || 0 }}%
                </v-chip>
                <v-chip
                  :color="getCPUUsageColor(metrics.cpu_usage)"
                >
                  CPU使用率: {{ metrics.cpu_usage || 0 }}%
                </v-chip>
              </v-card-text>
            </v-card>
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'AdminView',
  data() {
    return {
      metrics: null,
      error: null,
      refreshInterval: null
    }
  },
  async created() {
    await this.fetchMetrics()
    // 30秒ごとにメトリクスを更新
    this.refreshInterval = setInterval(this.fetchMetrics, 30000)
  },
  beforeUnmount() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval)
    }
  },
  methods: {
    async fetchMetrics() {
      try {
        const token = localStorage.getItem('token')
        const response = await axios.get('http://localhost:8080/api/admin/metrics', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })
        this.metrics = response.data
      } catch (error) {
        this.error = error.response?.data?.detail || 'メトリクスの取得に失敗しました'
        if (error.response?.status === 403) {
          this.$router.push('/')
        }
      }
    },
    formatDate(dateString) {
      return new Date(dateString).toLocaleString('ja-JP')
    },
    getActivityColor(type) {
      const colors = {
        'login': 'primary',
        'logout': 'grey',
        'chat': 'success',
        'error': 'error',
        'warning': 'warning'
      }
      return colors[type] || 'primary'
    },
    getMemoryUsageColor(usage) {
      if (!usage) return 'grey'
      if (usage < 50) return 'success'
      if (usage < 80) return 'warning'
      return 'error'
    },
    getCPUUsageColor(usage) {
      if (!usage) return 'grey'
      if (usage < 30) return 'success'
      if (usage < 70) return 'warning'
      return 'error'
    }
  }
}
</script>

<style scoped>
.admin-container {
  padding: 20px;
}

.text-caption {
  font-size: 0.875rem;
  color: rgba(0, 0, 0, 0.6);
}
</style> 