import api from './api';
import { User, UserProfile, UserSettings } from '@/types/user';

const userService = {
  // Get user profile
  getProfile: async (): Promise<UserProfile> => {
    try {
      const response = await api.get<UserProfile>('/users/profile');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Update user profile
  updateProfile: async (data: Partial<UserProfile>): Promise<UserProfile> => {
    try {
      const response = await api.put<UserProfile>('/users/profile', data);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Update user avatar
  updateAvatar: async (file: File): Promise<{avatarUrl: string}> => {
    try {
      const formData = new FormData();
      formData.append('avatar', file);

      const response = await api.post<{avatarUrl: string}>('/users/avatar', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Update user settings
  updateSettings: async (settings: UserSettings): Promise<UserProfile> => {
    try {
      const response = await api.put<UserProfile>('/users/settings', { settings });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get user activity
  getUserActivity: async (): Promise<{
    conversations: number;
    applications: number;
    lastActive: Date;
  }> => {
    try {
      const response = await api.get('/users/activity');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get user statistics
  getUserStatistics: async (): Promise<{
    twinCount: number;
    messageCount: number;
    interviewCount: number;
    applicationCount: number;
  }> => {
    try {
      const response = await api.get('/users/statistics');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Delete user account
  deleteAccount: async (password: string): Promise<void> => {
    try {
      await api.post('/users/delete-account', { password });
    } catch (error) {
      throw error;
    }
  },
};

export default userService; 