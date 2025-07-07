import { useContext } from 'react';
import { UserContext } from '@/contexts/UserContext';
import { UserProfile } from '@/types/user';

/**
 * Custom hook for user profile functionality
 * @returns User profile state and methods
 */
export const useUser = () => {
  const context = useContext(UserContext);
  
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  
  /**
   * Update user profile information
   */
  const updateProfile = async (data: Partial<UserProfile>) => {
    try {
      await context.updateProfile(data);
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : '更新资料失败'
      };
    }
  };
  
  /**
   * Update user settings
   */
  const updateSettings = async (settings: UserProfile['settings']) => {
    try {
      if (!settings) {
        return { success: false, error: '设置不能为空' };
      }
      
      await context.updateSettings(settings);
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : '更新设置失败'
      };
    }
  };
  
  /**
   * Get user's full name
   */
  const getFullName = () => {
    if (!context.profile) return '';
    return `${context.profile.firstName} ${context.profile.lastName}`;
  };
  
  /**
   * Check if user has completed their profile
   */
  const isProfileComplete = () => {
    if (!context.profile) return false;
    
    const { firstName, lastName, email } = context.profile;
    return Boolean(firstName && lastName && email);
  };
  
  return {
    profile: context.profile,
    isLoading: context.isLoading,
    error: context.error,
    updateProfile,
    updateSettings,
    getFullName,
    isProfileComplete,
  };
}; 