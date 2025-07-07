import { User } from './user';

export interface DigitalTwin {
  id: string;
  userId: string;
  name: string;
  description?: string;
  avatar?: string;
  createdAt: Date;
  updatedAt: Date;
  isPublic: boolean;
  status: TwinStatus;
  trainingStatus: TrainingStatus;
  communicationStyle?: CommunicationStyle;
  knowledgeSourceIds: string[];
  skillLevel: number;
  configurations?: TwinConfiguration;
}

export enum TwinStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  PENDING = 'pending',
  SUSPENDED = 'suspended',
}

export enum TrainingStatus {
  NOT_STARTED = 'not_started',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

export enum CommunicationStyle {
  PROFESSIONAL = 'professional',
  FRIENDLY = 'friendly',
  CONCISE = 'concise',
  DETAILED = 'detailed',
  ENTHUSIASTIC = 'enthusiastic',
}

export interface TwinConfiguration {
  responseLength?: 'short' | 'medium' | 'long';
  knowledgeDepth?: 'basic' | 'intermediate' | 'expert';
  availabilityHours?: {
    start: string;
    end: string;
  };
  automaticReplies?: boolean;
  aiModel?: string;
}

export interface KnowledgeSource {
  id: string;
  twinId: string;
  type: KnowledgeSourceType;
  name: string;
  content?: string;
  url?: string;
  fileUrl?: string;
  status: KnowledgeSourceStatus;
  createdAt: Date;
  updatedAt: Date;
  extractedData?: any;
}

export enum KnowledgeSourceType {
  RESUME = 'resume',
  LINKEDIN = 'linkedin',
  GITHUB = 'github',
  BLOG = 'blog',
  PROJECT = 'project',
  DOCUMENT = 'document',
  MANUAL_ENTRY = 'manual_entry',
  OTHER = 'other',
}

export enum KnowledgeSourceStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

export interface TwinConversation {
  id: string;
  twinId: string;
  userId: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
  lastMessageAt: Date;
  messages: TwinMessage[];
}

export interface TwinMessage {
  id: string;
  conversationId: string;
  content: string;
  sender: 'user' | 'twin';
  timestamp: Date;
  status: 'sending' | 'sent' | 'delivered' | 'read' | 'failed';
}

export interface TwinSkill {
  id: string;
  twinId: string;
  name: string;
  category: string;
  level: number; // 1-5 scale
  confidence: number; // 0-100%
  sources: string[]; // References to knowledge sources
}

export interface TwinEvaluation {
  id: string;
  twinId: string;
  userId: string;
  score: number; // 1-5 scale
  feedback?: string;
  createdAt: Date;
}

export interface TwinState {
  twin: DigitalTwin | null;
  isLoading: boolean;
  error: string | null;
  conversations: TwinConversation[];
  currentConversation: TwinConversation | null;
  knowledgeSources: KnowledgeSource[];
  skills: TwinSkill[];
} 