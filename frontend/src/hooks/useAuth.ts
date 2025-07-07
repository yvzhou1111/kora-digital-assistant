import { useContext } from 'react';
import { AuthContext } from '@/contexts/AuthContext';
import { LoginCredentials, RegisterData } from '@/types/user';

/**
 * Custom hook for authentication functionality
 * @returns Authentication state and methods
 */
export const useAuth = () => {
  const context = useContext(AuthContext);
  
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  
  /**
   * Login with email and password
   */
  const login = async (credentials: LoginCredentials) => {
    try {
      await context.login(credentials.email, credentials.password);
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : '登录失败'
      };
    }
  };
  
  /**
   * Register a new user
   */
  const register = async (data: RegisterData) => {
    try {
      if (data.password !== data.confirmPassword) {
        return { success: false, error: '两次输入的密码不一致' };
      }
      
      await context.register(
        {
          email: data.email,
          firstName: data.firstName,
          lastName: data.lastName,
          displayName: `${data.firstName} ${data.lastName}`,
          role: data.role,
        }, 
        data.password
      );
      
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : '注册失败'
      };
    }
  };
  
  /**
   * Logout the current user
   */
  const logout = () => {
    context.logout();
  };
  
  return {
    isAuthenticated: context.isAuthenticated,
    user: context.user,
    isLoading: context.isLoading,
    error: context.error,
    login,
    register,
    logout,
  };
}; 